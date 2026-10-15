# Scripts

Maintenance, validation, and figure-generation helpers for `doe-testbed`.
None of these are part of the Python package; they are CLI utilities
invoked from the repo root with `python scripts/<name>.py`.

## One-click pipeline (Stage 0 → 10)

The DOE-Benchmark experimental pipeline is wired into a single orchestrator.
Use a local machine for smoke checks, unit tests, static audits and documentation
work. The headline sweep + D1/D2/D3 + figures + paper macros require the Linux
server resources described in the [reproducibility guide](../docs/REPRODUCIBILITY.md).

| Entry point | Platform | Purpose |
|---|---|---|
| [`run_pipeline.py`](run_pipeline.py) | any | Python orchestrator; resolves stage commands, runs them as subprocesses, verifies expected outputs after each stage, forces BLAS/OpenMP thread caps to 1 before child processes, logs to `logs/pipeline_<UTC>.log`; the log is authoritative and terminal/stdout streaming is best-effort so a broken SSH pipe does not invalidate a completed stage. |
| [`run_pipeline.ps1`](run_pipeline.ps1) | Windows (pwsh 7+) | Smoke/preflight wrapper only: locates Python, pins BLAS threads, and refuses local full-sweep launches unless the command is a dry-run or Stage-1 preflight. |
| [`run_pipeline.sh`](run_pipeline.sh) | Linux / macOS / WSL | Same as `.ps1` but for `bash`; also the right entry point for `tmux` / `nohup`. The wrapper overrides inherited BLAS/OpenMP thread variables to 1 before Python starts. Operators may set positive `N_JOBS`, `BO_N_JOBS`, `MEM_CAP_GB`, `TIMEOUT_S`, `BO_TIMEOUT_S`, `REF_TIMEOUT_S`, `HEARTBEAT_S`, or `MAX_WALL_CLOCK_H`; the wrapper injects them only when the matching CLI flag is absent, so explicit CLI flags remain authoritative and the final values are recorded by command history plus `run_config.json`. |

### Common usage

```pwsh
# Smoke pipeline: Stage 0 writes _smoke_gate; audited 10-method sweep + report writes _smoke:
pwsh -File scripts/run_pipeline.ps1 -Profile smoke
# Full-profile no-sweep preflight on Windows:
pwsh -File scripts/run_pipeline.ps1 -Profile full -ToStage 1
# Minimal runner gate only:
python scripts/run_pipeline.py --profile smoke --only smoke
# Server/operator mode only: full run with an isolated smoke preflight:
python scripts/run_pipeline.py --profile full --with-smoke --confirm-full-run

# Full publication sweep (Linux server only, >=32 core / >=64 GB RAM, ~30-50 h).
# By default this repeats strict clean-sweep preflight, then runs Stage 1-10;
# add --with-smoke if you also want Stage 0:
bash scripts/run_pipeline.sh

# Explicit server tuning without editing source files. CLI flags still win.
N_JOBS=24 BO_N_JOBS=4 MEM_CAP_GB=96 bash scripts/run_pipeline.sh

# Server/archive mode: re-run only analysis + figures + macros after Stage 2 exists:
python scripts/run_pipeline.py --profile full --from-stage 3 --confirm-full-run

# Run only the figdata + figure stages after editing prepare_figdata.py:
python scripts/run_pipeline.py --profile full --only figdata figures --confirm-full-run

# Verify what would run without spawning any subprocess:
pwsh -File scripts/run_pipeline.ps1 -Profile full -DryRun
python scripts/run_pipeline.py --profile full --confirm-full-run --dry-run
```

The pipeline always pre-flights against `paper/preregistration.md`
(`validate_run_config.py`, Stage 1) and refuses to launch on protocol
deviation unless `--allow-deviation` is explicit. See the module
docstring of `run_pipeline.py` for the long-form rationale.

## Categories

### A. Routine validation (run before every release)

| Script | Purpose | Typical runtime |
|---|---|---|
| [`check_local_pre_server_ready.py`](check_local_pre_server_ready.py) | One-command local no-sweep checklist before server handoff: doc links, case source/note fingerprints, case-validation + preregistration tests, pipeline-orchestrator tests, sweep-integrity audit contract tests, server-operator packaging/return-contract tests, repository `.gitattributes` LF/binary portability contract, the NeurIPS 2026 Evaluations & Datasets-style benchmark/reusable-artifact audit contract in `docs/EXPERIMENT_BENCHMARK_AUDIT.md`, D1 pairing-mode tests, targeted case-physics tests, and pre-results paper draft gate. | ~2-5 min |
| [`check_doc_links.py`](check_doc_links.py) | Verify every relative link in `docs/*.md`, `paper/*.md`, `README*.md` and `scripts/README.md` resolves to a file in the repo. | <1 s |
| [`validate_run_config.py`](validate_run_config.py) | Pre-flight gate against `paper/preregistration.md` (pool / budgets / n-repeats / seed-strategy / hyperparameters consistency). | <1 s |
| [`generate_case_validation_summary.py`](generate_case_validation_summary.py) | Regenerate `docs/CASE_VALIDATION_SUMMARY.md` from the case-validation ledger, the `案例正确性校验/` review folders, and the supporting `tests/_cross_validation/` inventory; strict mode also checks `docs/case_validation_note_sources.csv` is current. | Stage 10 of the pipeline; rerun after case ledger, note-source, validation-folder, or cross-validation inventory changes. |
| [`update_case_validation_sources.py`](update_case_validation_sources.py) | Verify or refresh `docs/case_validation_sources.csv`, which pins the case-validation ledger to current case source hashes. | <1 s |
| [`update_case_validation_note_sources.py`](update_case_validation_note_sources.py) | Verify or refresh `docs/case_validation_note_sources.csv`, which pins human validation notes and reference attachments to path/size/SHA-256 fingerprints. | ~10 s |
| [`audit_case_experiment_readiness.py`](audit_case_experiment_readiness.py) | Batch-evaluate every registered case on lower/upper/centre/random points plus deterministic physical envelope probes and write `docs/case_experiment_readiness.csv`; catches shape, NaN/Inf, in-place input mutation, repeat-evaluate non-determinism, `expected_range` metadata drift, failure-threshold/topology/PDE/reference metadata drift, and smoke timing surprises before a long sweep. | ~10 s |
| [`audit_publication_readiness.py`](audit_publication_readiness.py) | Roll up case-validation, case-readiness, latest smoke sweep, full-sweep, server-return package, draft-paper, and submission gates into `docs/PUBLICATION_READINESS_AUDIT.md`; the full-sweep roll-up revalidates the current `results.csv` row-level contract in addition to manifest hashes/summaries; `--launch-preflight` checks only the pre-full-sweep gates that must pass before the server run and writes `docs/PUBLICATION_LAUNCH_PREFLIGHT.md` by default; accepts `--logs-dir` when returned command history is staged outside `logs/`. | <1 min |
| [`audit_macros.py`](audit_macros.py) | Static check that every `\Num*` token used in `nature_cs_main.tex` has a definition in `paper/numbers.tex`. | <1 s |
| [`check_paper_draft_ready.py`](check_paper_draft_ready.py) | Positive gate for the pre-results Nature-family draft package: required manuscript files exist, pending-result markers are present, the D1/D2/D3 decision ledger keeps its sign-off structure, SI exists, `si_cases.*` case evidence tiers match `docs/case_validation_status.csv`, figures are referenced, cover letter length/structure is editor-ready, `numbers.tex` macros match `numbers.json`, and `figure_manifest.json` hashes match local figures + figdata. | <1 s |
| [`check_submission_ready.py`](check_submission_ready.py) | Final submission gate: rejects pending markers, `TBD` / `XXX`, placeholder DOIs, stale numbers, duplicate or mismatched `numbers.tex` macros, incomplete manifests, manifest artifact symlinks, current `results.csv` row-level contract drift, missing files, editorial-summary/cover-letter format drift, incomplete signed D1/D2/D3 decision ledger semantics, stale case validation/readiness ledgers, stale validation-note fingerprints, stale SI case evidence tiers, missing multiplicity/D2-policy SI notes, missing D1/D2/D3 decision artifacts plus MC-noise analysis, stale `numbers.json` input provenance, `numbers.json`/`figdata_manifest.json` results-CSV SHA drift from `RUN_MANIFEST.json`, figdata analysis-policy drift from the preregistered D1/D2/D3/Fig. 6 primary policies, archived Python lock drift from the Stage-2 `run_config.json`, `RUN_MANIFEST.json` temp/scratch provenance drift from the Stage-2 `run_config.json` resolved path and `TMPDIR`/`SLURM_TMPDIR`/`SCRATCH`-style env vars, malformed root `.zenodo.json`/`codemeta.json`/`croissant-metadata.json`/`ro-crate-metadata.json` metadata, and CVD-unsafe figure palette drift. | <1 s |
| [`cross_validate_sobol.py`](cross_validate_sobol.py) | Independently re-estimate every stored Sobol index using SALib at a different sample size + seed. | ~5 min (default `N=4096`) |
| [`check_optional_dependencies.py`](check_optional_dependencies.py) | Pre-flight optional-dependency check before long runs; the formal sweep passes `--require-botorch`, `--require-formal-case-backends`, and `--require-runtime-monitoring` so the 20-method headline protocol and formal case backends cannot silently degrade. | <10 s |
| [`run_case_physics_tests.py`](run_case_physics_tests.py) | Run the targeted per-case physics test files (`tests/test_*_physics.py` plus the expected-range envelope tests) and verify their coverage against `docs/case_validation_status.csv`. | ~1-2 min |

`audit_publication_readiness.py` intentionally exits non-zero while the server
full sweep is blocked.  The server-return package and final submission gates
are marked `DEFERRED` until the full sweep passes; read the generated Markdown
status report before deciding whether action is needed.

The local and server preflight wrappers import their server-operator pytest
nodeids from [`server_operator_contracts.py`](server_operator_contracts.py).
Update that shared module, rather than editing two separate command lists, when
adding launch, monitor, return-package, manifest, or submission-readiness
contracts that must be enforced before the Linux full sweep.

### B. Figure / artifact generation

| Script | Output | When to run |
|---|---|---|
| [`prepare_figdata.py`](prepare_figdata.py) | `paper/figdata/*.csv` plus `figdata_manifest.json` — the figdata layer consumed by `plots_nature.py`, including input hashes, a machine-readable row/status/failure-category `results_summary`, and the machine-checkable D1/D2/D3/Fig. 6 analysis-policy contract. | Stage 8 of the pipeline (after D1–D3 outputs are fresh). |
| [`plots_nature.py`](plots_nature.py) | `paper/figures/{pdf,png}/fig{1..6}.{pdf,png}` (Nature-format main figs); `figure_manifest.json` records figure hashes, source-figdata hashes, the linked `figdata_manifest.json` hash, and the copied D1/D2/D3/Fig. 6 analysis-policy contract; partial `--fig` selections are marked and rejected before submission. | Stage 9 of the pipeline. |
| [`../benchmark/scripts/validate_test_protocol.py`](../benchmark/scripts/validate_test_protocol.py) | `benchmark/results/full_benchmark/analysis/test_set_sufficiency.csv` plus cross-metric and robustness diagnostics consumed by paper numbers and submission checks. | Stage 10a.0 of the pipeline, before `fill_paper_numbers.py`. |
| [`fill_paper_numbers.py`](fill_paper_numbers.py) | `paper/numbers.{tex,json}` — every `\Num*` macro for the manuscript, with `numbers.json` input provenance hashes for the restored sweep, D1/D2/D3, and test-protocol sources. | Stage 10a of the pipeline. |
| [`generate_si.py`](generate_si.py) | `paper/si_*.{tex,md}` — baseline SI tables auto-derived from the canonical case + hyperparameter registries, plus the multiple-testing and D2 exploratory-denominator policy notes. | Stage 10a.5 of the pipeline; rerun after method registry, case metadata, or preregistered analysis-policy changes. |
| [`build_supplementary.py`](build_supplementary.py) | `paper/si_*_clean.tex` and `paper/si_d3_per_method.tex` — the LaTeX table fragments consumed by `paper/supplementary.tex`, regenerated from committed analysis artifacts only (no hand-entered numbers). | Rerun after `generate_si.py` outputs or the reliability/Sobol analysis artifacts change, before recompiling `supplementary.tex`. |
| [`derive_pooled_reliability_stats.py`](derive_pooled_reliability_stats.py) | `benchmark/results/reliability_recovery/pooled_reliability_summary.json` — the pooled D3 signed-bias estimate, its case-cluster bootstrap CI, the one-sided Wilcoxon p-value, and the D2 Clopper--Pearson count-test bound quoted in the manuscript, recomputed from the frozen sweep artifacts. | Rerun to re-derive the pooled manuscript headline statistics; read-only over the archive. |
| [`update_case_validation_note_sources.py`](update_case_validation_note_sources.py) | `docs/case_validation_note_sources.csv` — fingerprints for human validation notes and reference attachments. | Stage 10a.6 of the pipeline. |
| [`generate_case_validation_summary.py`](generate_case_validation_summary.py) | `docs/CASE_VALIDATION_SUMMARY.md` — materialized case-validation summary tied to the ledger and note-source fingerprints. | Stage 10a.7 of the pipeline. |
| [`generate_run_manifest.py`](generate_run_manifest.py) | `RUN_MANIFEST.json` plus `benchmark/results/full_benchmark/python_environment_pip_freeze.txt` — immutable snapshot (git, env, full archived `pip freeze --all` lock, physical/effective RAM, cgroup memory limits, process `ulimit` values, resolved temp/scratch storage context including `TMPDIR`/`SLURM_TMPDIR`/`SCRATCH`-style env vars plus resolved path/writability/free space, where writability is confirmed by a real create/write/flush/fsync/delete probe rather than permission bits alone, `results.csv`, fresh `sweep_integrity_report.md`, resolved case registry, command-history `label<TAB>command` entries, raw CSV header/ragged-row contract counters such as `n_csv_contract_errors`, row-contract counters including missing protocol keys, peak-RSS resource roll-ups, symlink markers, and sha256 of every real archived artefact). | Stage 10b of the pipeline, after `sweep_audit`. |
| [`check_submission_ready.py`](check_submission_ready.py) | No new file — final gate over paper sources, manifest, numbers, figures, figdata input hashes and analysis policy, D1/D2/D3 decision artifacts, MC-noise analysis, case validation/readiness ledgers, validation-note fingerprints, root FAIR metadata, and CVD-safe palette contract. | Final Stage 10 gate. |
| [`generate_paper_figures.py`](generate_paper_figures.py) | `paper/fig/*.pdf` (Sobol bar charts + 2-D response surfaces). | Before re-rendering `paper/nature_cs_main.tex`. |
| [`generate_sobol_charts.py`](generate_sobol_charts.py) | `docs/img/sobol/*.png` (per-case Sₜ bar charts). | Before publishing the docs site. |

The analysis entry points that consume an existing full-sweep `results.csv`
reuse the Stage-8 row-level result contract before writing derivative
artifacts.  This includes D1 `run_ranking_reversal.py --reuse-from-csv`, D2
`run_sample_complexity.py`, D3 `run_reliability_recovery.py`,
`validate_test_protocol.py`, `prepare_figdata.py`, and
`fill_paper_numbers.py --require-results-contract`.  Formal D2, D3,
test-protocol, figdata, and paper-number commands also pass
`--require-headline-source`, which requires the canonical
`benchmark/results/full_benchmark/results.csv` path and the exact frozen
34-case × 20-`nature_pool()`-method × 4-budget × 30-repeat grid before
analysis, figdata, or numbers are written.  Manually running a later stage
therefore still rejects duplicate `(case_id, method, budget_mult, repeat)`
cells, blank protocol keys, invalid statuses, bad successful-row `nrmse`,
non-ok rows with populated performance metrics, missing failure reasons,
invalid timing, `t_total_s`/runtime-timestamp duration disagreement,
non-headline result paths, and
SOTA/MF method contamination before stale or corrupt analysis can enter the
paper package.  The final submission gate also requires the preregistered D1
pairing-mode / budget combinations to have primary `complete_case` records in
both `correlation.json` and each `pairing_*/rank_comparison.csv`; the numbers
stage also prefers that primary record when extracting D1 macros and prefers
primary `penalized_failed_in_denominator` rows for D2 macros, protecting paper
numbers from non-primary rank-transfer or sample-complexity rows.  For D3, the
numbers stage reconstructs Holm values from raw two-sided p-values when needed
but leaves D3 headline macros as `N/A` if neither source of p-values exists,
instead of reporting a spurious "No Bias".  The
publication-readiness roll-up and final submission gate repeat the row-level
results check against the current local full-sweep CSV through the
`RUN_MANIFEST.json` result contract, including the manifest's
`n_missing_protocol_key_values` counter and observed case/method set-count
closure.  They also recompute the current CSV's row count, observed
case/method/budget/repeat sets, and status counts and compare those values
back to `RUN_MANIFEST.json`, so updating manifest hashes or summary fields
alone cannot make a corrupted or swapped CSV publication-ready or
submission-ready.

The Stage-2 `--skip-existing` resume path is fail-closed before any cell is
skipped: an unreadable or unparseable `results.csv`, symlinked CSV path, stale
schema, header-only CSV with no completed data rows, duplicate/blank raw CSV
header names, ragged raw CSV rows whose field count differs from the header,
source-hash drift, protocol-grid drift, duplicate cell, blank protocol
key, invalid status, bad successful-row metric, missing failure reason, or
non-ok row with populated performance metrics, invalid timing, including
disagreement between `t_total_s` and
`cell_finished_utc - cell_started_utc`, aborts the resume.  The streamed
on-disk `results.csv` is the canonical Stage-2 artifact: after a resume the
in-memory `BenchmarkResult` contains only newly computed rows, so
`run_full_benchmark.py` must report the streamed file and must not overwrite it
from memory. Use a clean output directory or explicitly repair the CSV instead
of appending new rows to a corrupt archive.

### C. One-shot audits / surveys (kept for reproducibility)

These were used to author specific documentation pages; they are short
read-only scripts that can be re-run any time to sanity-check the
metadata layer.

| Script | Used to author |
|---|---|
| [`audit_effective_dim.py`](audit_effective_dim.py) | Cross-checks `meta.effective_dim` against the count of variables with $S_T \ge 0.05$. |
| [`survey_failure_thresholds.py`](survey_failure_thresholds.py) | Initial Markdown table for `docs/failure_thresholds.md`. |
| [`survey_sobol_outputs.py`](survey_sobol_outputs.py) | Detects which non-default output index each case used to record its Sobol indices. |
| [`inspect_failure_conditions.py`](inspect_failure_conditions.py) | Prints every case's `failure_condition` string and per-output threshold metadata. |
| [`build_zh_overview.py`](build_zh_overview.py) | Regenerates the expanded Chinese project overview (`PROJECT_OVERVIEW_zh.md`) from the archived analysis artifacts, so no number in the overview is hand-entered. |

### D. Sweep launchers and helpers

Low-level launchers for server operators and post-registration supplementary
experiments.  Most operators should call `run_pipeline.{py,ps1,sh}` instead;
the launchers below are kept for fine-grained restart, tmux-style production
workflows, or legacy constrained-host diagnostics.  The Nature CS headline
archive should be produced by the frozen full pipeline on the designated Linux
server, not by a local review workstation launcher.

| Script | Purpose |
|---|---|
| [`launch_sweep_remote.sh`](launch_sweep_remote.sh) | Linux Stage-2 launcher retained for fine-grained server restart workflows; validates all-case experiment readiness, auto-tunes `N_JOBS`, `BO_N_JOBS`, and `MEM_CAP_GB` from the effective RAM after current-process cgroup memory limits are considered, and checks any existing headline `results.csv` with the runtime health gate plus `--validate-current-environment` before dispatch so resume is blocked if the active interpreter/package/thread-cap environment drifted from the Stage-2 startup snapshot; prefer `run_pipeline.sh` for the formal full archive. |
| [`launch_sweep_local_16gb.ps1`](launch_sweep_local_16gb.ps1) | Legacy constrained Windows Stage-2 launcher for diagnostics only; not the Nature CS headline run path. |
| [`launch_sota_robustness.sh`](launch_sota_robustness.sh) | Amendment A1 — SOTA pool (TURBO + D-optimal), isolated under `benchmark/results/sota_robustness/` and guarded against writing to or reusing the wrong headline directory. |
| [`launch_mfbo.sh`](launch_mfbo.sh) | Amendment A2 — multi-fidelity supplement, isolated under `benchmark/results/mf_supplementary/`, guarded against writing to headline/SOTA directories, and default-capped to `N_JOBS=5` unless the operator explicitly overrides it. |
| [`launch_d1_pairings.sh`](launch_d1_pairings.sh) | Amendment A3 — D1 pairing-mode robustness restart/diagnostic launcher; matches formal Stage 4 (`pool=nature`, 20 methods, `pairing-mode all`) and writes only to `benchmark/results/ranking_reversal/`. |
| [`_remote_lib.sh`](_remote_lib.sh) | Shared bash helper sourced by the four launchers above; auto-tunes `N_JOBS`, `BO_N_JOBS`, and `MEM_CAP_GB` from the effective RAM after current-process cgroup memory limits are considered while keeping BoTorch parallelism at a positive integer even on undersized diagnostic hosts, rejects non-positive/invalid environment overrides, rejects unknown persistence modes, starts tmux sessions from the current repo directory, and refuses launch when the PID file still points to a live process. |
| [`check_server_launch_ready.py`](check_server_launch_ready.py) | No-sweep server launch preflight: case fingerprints, 20-method `nature_pool()` cardinality, formal case backend imports such as NeuralFoil/AeroSandbox, runtime monitoring imports such as `psutil`, effective-RAM and disk-capacity checks on Linux including cgroup memory limits, the currently resolved temp/scratch storage path and `TMPDIR`/`SLURM_TMPDIR`/`SCRATCH`-style overrides, an actual temp-directory create/write/flush/fsync/delete probe for the current job, runtime-row contracts, the full sweep-integrity audit contract, server-operator packaging/return-contract tests, the NeurIPS 2026 Evaluations & Datasets-style benchmark/reusable-artifact audit contract in `docs/EXPERIMENT_BENCHMARK_AUDIT.md`, targeted physics tests, P2 web/literature audit coverage and evidence-tier consistency with the case-validation ledger, official-source snapshot markers for high-risk P2 aerodynamics/geomechanics/multidisciplinary/analytical-control cases, LF line-ending checks plus `bash -n` syntax checks for server-facing shell wrappers when a usable bash is available, an explicit Windows skip when only an unusable WSL launcher is present, Stage 1 gates, publication launch-preflight, full command-graph dry-run, required-stage graph validation, existing `results.csv` resume-health screening, clean-sweep detection for stale `run_config.json`, `python_environment_pip_freeze.txt`, reports, manifests, and analysis artifacts before a fresh first launch, per-subprocess timeout/elapsed-time reporting via `--check-timeout-s` or `DOE_BENCHMARK_LAUNCH_CHECK_TIMEOUT_S`, and a runtime safety check that rejects sweep-launch commands. |
| [`_check_sweep_progress.py`](_check_sweep_progress.py) | Parse the headline `results.csv` or an explicit CSV path mid-sweep to print remaining-cells / ETA. |
| [`check_sweep_runtime_health.py`](check_sweep_runtime_health.py) | Read-only runtime health gate for the active headline sweep or an explicit returned/staged CSV path; defaults to the headline protocol (`n_test=1000`, `test_seed=0`, `case_repeat`, `seed_start=1000`) while allowing explicit smoke/diagnostic protocol overrides; can bind partial rows to the Stage-2 `run_config.json` startup snapshot automatically when it sits next to `results.csv`, or require it with `--require-run-config` / pass it explicitly with `--run-config-json`; during the missing/header-only startup window, `--allow-missing` stays non-blocking for first-row latency but still validates any explicitly requested `run_config.json` (the remote monitor passes the sibling startup snapshot explicitly once it exists); `--validate-current-environment` further compares the current interpreter, required package versions, BLAS thread caps, and Torch deterministic/TF32/cuDNN backend state against the recorded startup snapshot for same-host resume/packaging checks; the gate flags symlinked or malformed CSVs, duplicate/blank raw CSV headers, ragged raw CSV rows, protocol drift with field-specific `case_id`/`method`/`budget_mult`/`repeat` diagnostics, duplicate protocol cells, stale updates, bad statuses, invalid `ok`-row `nrmse`, timeout/failed rows with populated performance metrics, `n_test`/`test_seed`/per-cell `seed` drift, missing/invalid row-level runtime provenance (`worker_host`, `worker_pid`, UTC cell timestamps, RSS start/end/peak trace), missing or malformed `run_config.json` runtime controls, runtime-environment snapshot, Torch reproducibility-policy snapshot, and temp/scratch storage snapshot (`resolved_path`, `exists`, `writable`, `free_gb`, plus `TMPDIR`/`SLURM_TMPDIR`/`SCRATCH`-style env vars), required formal-run packages recorded as anything other than `installed`, disagreement between `t_total_s` and the recorded UTC cell duration, high timeout/failed fractions, RSS close to the configured memory cap, low free disk space near either `results.csv` or the current temp/scratch path when `--disk-warning-gb` is set, and invalid threshold settings without launching experiments; its shared failure taxonomy covers OOM/timeout/dependency/backend/numerical/worker-crash/scheduler/storage strings including `ExitCode=0:9`, `signal: killed`, `DUE TO PREEMPTION`, `Transport endpoint is not connected`, `ESTALE`, and `Too many open files`; its report also emits `[summary]` lines for failure-category counts, RSS peak telemetry, and `t_total_s` wall-time telemetry, plus `[artifact]` fingerprints for both `results.csv` and `run_config.json`, so OOM/crash/timeout/resource/long-tail runtime evidence and result/config provenance are visible in the archived return package without reparsing `results.csv`. |
| [`audit_sweep_integrity.py`](audit_sweep_integrity.py) | Final Stage-2b row-contract gate for the formal headline `results.csv`; checks the complete case/method/budget/repeat grid, source fingerprints, row-level protocol digests, seeds, statuses, failure taxonomy, metrics, timing, UTC/RSS runtime trace, duplicate cells, raw CSV structure, and success/failure fractions. The formal pipeline passes `--run-config-json benchmark/results/full_benchmark/run_config.json --require-run-config`, binding the completed archive to the atomic Stage-2 startup snapshot before any report, figdata, SI, numbers, manifest, or submission-ready stage can consume it. |
| [`sweep_progress_breakdown.py`](sweep_progress_breakdown.py) | Per-method / per-budget breakdown of completion + status; handles missing, header-only, malformed, or schema-incomplete `results.csv` without traceback and points operators to the formal runtime health gate. |
| [`check_progress_remote.sh`](check_progress_remote.sh) | Tail the headline sweep log over SSH and include the read-only runtime health gate; forwards `MEM_CAP_GB`, `DISK_WARNING_GB`, `STALE_AFTER_MIN`, and `FAIL_ON_STALE` to the health gate so one watch command can surface memory pressure, low free-space risk, and stale-writer failures during unattended server runs; setting any non-empty `VALIDATE_CURRENT_ENV` additionally forwards `--validate-current-environment` so same-host monitoring can actively compare the current interpreter/package/thread-cap environment against the Stage-2 startup snapshot; one-shot mode returns non-zero when `sweep.pid` is dead while the headline sweep is incomplete, while watch mode keeps refreshing. |
| [`check_server_return_package.py`](check_server_return_package.py) | Local post-return package validator for the transferred tarball plus extracted full-sweep archive, root machine-readable metadata (`.zenodo.json`, `CITATION.cff`, `codemeta.json`, `croissant-metadata.json`, `ro-crate-metadata.json`), `run_config.json`, manifest, D1/D2/D3, figdata, figures, numbers, command history, runtime-health report, case-validation evidence files, benchmark-protocol audit, and statistical analysis plan; optional `--package server_return_package_<UTC>.tar.gz` rejects symlinked package/sidecar paths, verifies the `.sha256` sidecar digest and filename binding, rejects unsafe POSIX/Windows tar member paths, duplicate member paths, links, unsupported special members, required directory members are real directories, requires packaged `sweep_integrity_report.md` and `runtime_health_report.txt` to contain their PASS verdicts, requires the health report's `results.csv` logical path to still point at `benchmark/results/full_benchmark/results.csv`, requires the health report's `results.csv` size/SHA-256 fingerprint to match the packaged CSV, requires the health report's `run_config.json` logical path to still point at `benchmark/results/full_benchmark/run_config.json`, requires the health report's `run_config.json` size/SHA-256 fingerprint to match the packaged startup snapshot, requires proof that the health report checked `results.csv` against `run_config.json`, requires the `run_config.json runtime controls recorded and valid`, `run_config.json runtime environment recorded and valid`, `run_config.json torch reproducibility policy recorded and valid`, `current runtime environment matches run_config.json snapshot`, failure-category telemetry summary, runtime-resource telemetry summary, runtime-wall-time telemetry summary proofs from the current runtime-health gate, binds packaged `RUN_MANIFEST.json`, `paper/numbers.json`, and `paper/figdata/figdata_manifest.json` to the packaged `results.csv` SHA-256, checks packaged `logs/pipeline_command_history.txt` against the packaged `RUN_MANIFEST.json` command entries before extraction, recomputes the packaged `figdata_manifest.json` `results_summary` against the packaged `results.csv`, and rejects optional SOTA/MF robustness directories inside the headline package before the extracted tree is trusted.  The extracted-tree gate rejects symlinked base directories, rejects symlinked required files/directories, requires the root metadata files to be parseable and to contain key Zenodo/CodeMeta/Croissant/RO-Crate fields, requires Croissant `cr:FileObject` distributions and an extended `cr:RecordSet` mapping for task identity, budgets, seeds, protocol digest, status/failure category, primary metric, runtime/resource telemetry, and case/method source hashes back to concrete `results.csv` columns, requires the full-sweep `run_config.json` startup snapshot, rechecks that the archived `python_environment_pip_freeze.txt` lock still matches the recorded Stage-2 `run_config.json` core package versions, rechecks that `RUN_MANIFEST.json` machine temp/scratch provenance matches the Stage-2 `run_config.json` resolved path and `TMPDIR`/`SLURM_TMPDIR`/`SCRATCH`-style env vars, rechecks P2 web-validation audit evidence-tier markers against the case-validation ledger, requires official-source snapshot markers for high-risk P2 aerodynamics/geomechanics/multidisciplinary/analytical-control cases when those cases are present, verifies `pipeline_command_history.txt` uses `label<TAB>command` lines, matches well-formed manifest entries with exactly one non-dry-run formal full-pipeline `pipeline_invocation`, rejects bad manifest result-contract counters such as raw CSV structure errors via `n_csv_contract_errors` and missing protocol keys, requires the extracted full-sweep integrity and runtime-health reports to contain their PASS verdicts, requires the extracted health report to fingerprint the extracted `results.csv` and `run_config.json` and prove run-config binding plus runtime-control/runtime-environment/Torch-reproducibility/current-environment and telemetry-summary validation, catches stale `numbers.json` input provenance, and requires `numbers.json` plus `figdata_manifest.json` to bind their `results_csv` input SHA to the `RUN_MANIFEST.json` archived `results.csv` before manuscript sign-off. |
| [`create_server_return_package.py`](create_server_return_package.py) | Cross-platform packaging helper to run after the formal full pipeline; verifies required return artifacts are present/non-empty, including `.zenodo.json`, `CITATION.cff`, `codemeta.json`, `croissant-metadata.json`, `ro-crate-metadata.json`, the Stage-2 `run_config.json` startup snapshot, case-validation ledger/source fingerprints/readiness/summary/web-audit documents, `docs/EXPERIMENT_BENCHMARK_AUDIT.md`, and `docs/STATISTICAL_ANALYSIS_PLAN.md`, rechecks P2 web-validation audit evidence-tier consistency with the case-validation ledger, requires the Stage-2b `sweep_integrity_report.md` to contain `PASS: sweep satisfies the declared integrity gate.` and the final `runtime_health_report.txt` to contain the runtime-health PASS verdict plus a `results.csv` path that still resolves to the current headline archive, a canonical `benchmark/results/full_benchmark/results.csv` logical suffix, matching `results.csv` size/SHA-256 fingerprint, a `run_config.json` path that still resolves to the current Stage-2 startup snapshot, a canonical `benchmark/results/full_benchmark/run_config.json` logical suffix, matching `run_config.json` size/SHA-256 fingerprint, `run_config.json` binding proof, `run_config.json runtime controls recorded and valid` proof, `run_config.json runtime environment recorded and valid` proof, `run_config.json torch reproducibility policy recorded and valid` proof, `current runtime environment matches run_config.json snapshot` proof, failure-category telemetry summary, runtime-resource telemetry summary, and runtime-wall-time telemetry summary, rechecks that `benchmark/results/full_benchmark/python_environment_pip_freeze.txt` still matches the Stage-2 `run_config.json` core package versions before packaging, rechecks that `RUN_MANIFEST.json` temp/scratch provenance matches the Stage-2 `run_config.json` resolved path and `TMPDIR`/`SLURM_TMPDIR`/`SCRATCH`-style env vars before packaging, refuses symlinks/special members, requires the output filename to end with `.tar.gz`, rejects symlinked output paths, symlinked output parent directories, symlinked sidecar paths, and output paths that would overwrite packaged artifacts or self-include the tarball, writes a tar.gz using repo-relative paths, excludes SOTA/MF supplementary directories from the headline package, refuses to overwrite an existing output or sidecar unless requested, writes the tarball and sha256 sidecar atomically through a temporary directory, and self-validates the tarball before transfer back to the review workstation. |
| [`create_server_return_package.sh`](create_server_return_package.sh) | Linux server wrapper for `create_server_return_package.py`; resolves `$PYTHON`, `.venv/bin/python`, then `python3`/`python` so packaging uses the same environment as the server run; first writes `benchmark/results/full_benchmark/runtime_health_report.txt` by rerunning the read-only runtime health gate against the current `results.csv` and Stage-2 `run_config.json`, including current `results.csv` and `run_config.json` size/SHA-256 fingerprints plus the `--validate-current-environment` drift check, and writes that report through a temporary file that is cleaned on interruption before packaging only if that gate passes. |
| [`setup_remote.sh`](setup_remote.sh) | Bootstrap a fresh Linux host (venv + dependencies). |

## Conventions

Scripts in this folder share the following conventions; please follow
them for any new addition:

- Module docstring on line 1 stating purpose and any output paths.
- Imports in standard order: stdlib → third-party → `doe_testbed`.
- Output paths anchored at the repo root via
  `pathlib.Path(__file__).resolve().parent.parent`, never via `cwd()`.
- No dependency outside what `pip install -e ".[dev]"` provides
  (i.e., the `[project.optional-dependencies].dev` block in
  `pyproject.toml`).
- A shebang (`#!/usr/bin/env python`) is added when the script is
  intended to be executed directly on Linux / macOS; it is harmless
  on Windows.
- Type hints use `from __future__ import annotations` when the script
  has any.

## Adding a new script

1. Pick the right category (A/B/C/D) and add a row to the table above.
2. Follow the conventions above.
3. If the script is going to be run in CI, also wire it into
   `.github/workflows/ci.yml` and document the step in the relevant
   table.
