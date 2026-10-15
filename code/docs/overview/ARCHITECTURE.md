# Architecture — DOE-Benchmark

> **Audience.**  Anyone reading the repo for the first time.  By the end
> of this page you should know which file to open for any given task,
> what data flows where, and which directories are off-limits for
> editing.
>
> **Companion docs.**  This document explains *what is where*; for
> the current folder map and command entry points see
> [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md);
> *why* see [`docs/STATISTICAL_ANALYSIS_PLAN.md`](../STATISTICAL_ANALYSIS_PLAN.md);
> for *how to run it* see [`docs/REPRODUCIBILITY.md`](../REPRODUCIBILITY.md).

## 1. The 5-second view

```
            ┌─────────────────────┐
            │   doe_testbed/      │   ← 34 physics test cases
            │   (case library)    │     (LOCKED at v0.4.0-prereg)
            └──────────┬──────────┘
                       │  case.evaluate(X)
                       ▼
            ┌─────────────────────┐
            │   benchmark/        │   ← 20 DOE/surrogate/BO methods
            │   (runner + stats)  │     + protocol + statistics
            └──────────┬──────────┘
                       │  results.csv
                       ▼
            ┌─────────────────────┐
            │   scripts/          │   ← Stage 0–10 pipeline
            │   (orchestration)   │     + figdata + paper macros
            └──────────┬──────────┘
                       │  paper/figdata/*.csv,
                       │  paper/figures/*.pdf,
                       │  paper/numbers.{tex,json},
                       │  RUN_MANIFEST.json
                       ▼
            ┌─────────────────────┐
            │   paper/            │   ← Pre-registration + decision
            │   (protocol mirror) │     + submission templates
            └─────────────────────┘
```

The three **hard rules** of the layered architecture:

1. **`doe_testbed/` never imports from `benchmark/`** — the case library
   knows nothing about DOE methods.  Adding a new case must not change
   any benchmark code.
2. **`benchmark/` never reads `paper/`** — the experiment runner is
   *measurement-pure*; it writes `results.csv` and walks away.
3. **`scripts/` reads `benchmark/results/` and writes `paper/`** —
   never the reverse, never modifying upstream data.

These are enforced by the test suite (`benchmark/tests/test_smoke.py`,
`benchmark/tests/test_paper_pipeline.py`) and by the
`scripts/check_doc_links.py` audit.

## 2. Layer-by-layer

### 2.1 Library layer — `doe_testbed/`

Pure case library.  No DOE-method knowledge.

| Sub-package | Role |
|---|---|
| `doe_testbed/base/` | `Case`, `CaseMeta`, `OutputVariable` ABCs that every case implements. |
| `doe_testbed/cases/` | 34 case modules (one per physics problem); each defines `meta`, `evaluate`, `failure_threshold` if applicable. |
| `doe_testbed/registry/` | `list_cases()` / `get_case(id)` / `get_case_meta(id)` lookup. |
| `doe_testbed/utils/` | Sobol' / LHS sampling, normalisation, denormalisation helpers. |

**Public API**: `import doe_testbed as db; case = db.get_case(id); Y = case.evaluate(X_norm)`.  Frozen at v0.4.0-prereg per [`paper/preregistration.md`](../../../manuscripts/protocol/preregistration.md) §3.

### 2.2 Method + protocol layer — `benchmark/`

20 method adapters + the runner that drives them through the pre-registered protocol.  See [`benchmark/README.md`](../../benchmark/README.md) for the per-module breakdown.

Critical entry points:

| File | Public symbol | Role |
|---|---|---|
| `benchmark/__init__.py` | `BenchmarkRunner`, `derive_seed`, pool factories | The 40+ public-API names exposed by `import benchmark`. |
| `benchmark/core.py` | `BenchmarkRunner` | Orchestrates the (case × method × budget × repeat) sweep, streams `results.csv`, handles soft timeouts and `--skip-existing` resume. |
| `benchmark/methods.py` | `nature_pool()` (frozen 20), `sota_pool()`, `mf_pool()`, etc. | Method-pool factories; `nature_pool()` is the protocol-frozen 20 methods. |
| `benchmark/hyperparameters.py` | `HYPERPARAMETERS`, `assert_hyperparameters_consistent()` | Canonical HP table; CI-checked against the actual factory values. |
| `benchmark/reliability.py` | `reliability_metrics()` | Per-row P_f / abs-err / bal-acc / Brier; written into `results.csv` for the 16 cases with `failure_threshold`. |
| `benchmark/analysis.py` | `friedman_test`, `nemenyi_post_hoc`, `critical_difference` | Demšar (2006) Friedman+Nemenyi+CD; `studentized_range` fallback for k>20. |
| `benchmark/rigour.py` | `bootstrap_metric_ci`, `pareto_front`, etc. | Bootstrap CIs, Cliff's δ, Pareto front, fit-quality flags. |
| `benchmark/stratified.py` | `annotate_results`, `stratified_ranks`, `bootstrap_rank_shift_ci` | Per-topology / per-domain stratified analysis. |
| `benchmark/sample_complexity.py` | `compare_to_10d_rule()` | **Discovery 2** — n*/d per topology + 10·d rule. |
| `benchmark/reliability_recovery.py` | `per_method_reliability_summary()` | **Discovery 3** — signed P_f bias + Holm. |

Discovery script entry points are inside `benchmark/scripts/`:

| Script | Discovery |
|---|---|
| `benchmark/scripts/run_full_benchmark.py` | Stage 2 main sweep (writes `results.csv`). |
| `benchmark/scripts/run_ranking_reversal.py` | Stage 4a D1 — analytical vs physical with three pairings. |
| `benchmark/scripts/recompute_spearman_ci.py` | Stage 4b — bootstrap 95 % CI on Spearman ρ / Kendall τ. |
| `benchmark/scripts/run_sample_complexity.py` | Stage 5 D2. |
| `benchmark/scripts/run_reliability_recovery.py` | Stage 6 D3. |
| `benchmark/scripts/generate_report.py` | Stage 3 headline `REPORT.md`. |

### 2.3 Orchestration + figdata layer — `scripts/`

Top-level CLI tools that connect the runner outputs to paper artefacts.  The single most important entry point is the one-click pipeline.

| Script | Purpose |
|---|---|
| `scripts/run_pipeline.{py,ps1,sh}` | One-click Stage 0–10 orchestrator (Python + Windows + Linux). |
| `scripts/validate_run_config.py` | Pre-flight gate against `paper/preregistration.md`. |
| `scripts/prepare_figdata.py` | Stage 8 — emits `paper/figdata/fig{1..6}_*.csv`. |
| `scripts/plots_nature.py` | Stage 9 — Nature-format `paper/figures/{pdf,png}/fig{1..6}.{pdf,png}` (CVD-safe palette). |
| `scripts/fill_paper_numbers.py` | Stage 10a — emits `paper/numbers.{tex,json}` macros. |
| `scripts/generate_run_manifest.py` | Stage 10b — `RUN_MANIFEST.json` (git+env+sha256 archive). |
| `scripts/check_doc_links.py` | Static link-rot audit (covers all `*.md` under repo). |
| `scripts/audit_macros.py` / `scripts/check_submission_ready.py` | Submission-readiness audits. |
| `scripts/launch_*.{sh,ps1}` | Lower-level per-stage launchers (mostly superseded by `run_pipeline`). |

Full inventory in [`scripts/README.md`](../../scripts/README.md).

### 2.4 Protocol + submission layer — `paper/`

This directory keeps the **scientific protocol** plus **submission templates**.
Pipeline-generated manuscript artifacts are written back into `paper/`
(`paper/figdata/`, `paper/figures/`, `paper/numbers.{tex,json}`).

| File | Locked? | Role |
|---|---|---|
| `paper/preregistration.md` | yes (`v0.4.0-prereg`) | H₁/H₂/H₃, primary endpoints, decision rules, falsification clauses. |
| `paper/preregistration_amendments.md` | append-only | A1/A2/A3 post-freeze amendments. |
| `paper/preregistration_decisions.template.md` | template | Filled in **after** Stage 2 finishes; freezes H1/H2/H3 decision before any manuscript edit. |
| `paper/significance_statement.template.md` | template | 120/150/200-word editorial summary for the target sub-journal. |
| `paper/cover_letter.template.md` | template | ≤ 1-page sub-journal cover. |

### 2.5 Documentation layer — `docs/`

Reader-facing scientific methods, reproducibility and terminology documentation.

| File | Audience | Update trigger |
|---|---|---|
| [`docs/STATISTICAL_ANALYSIS_PLAN.md`](../STATISTICAL_ANALYSIS_PLAN.md) | "What does each Discovery test?" | Append-only post-freeze |
| [`docs/REPRODUCIBILITY.md`](../REPRODUCIBILITY.md) | "How do I reproduce this?" | After each pipeline change |
| [`docs/GLOSSARY.md`](../GLOSSARY.md) | "What does that acronym mean?" | When new term enters the project |
| [`docs/FAQ.md`](../FAQ.md) | Reviewers + replicators | After each Q&A from outside |

### 2.6 Test layer — `tests/` and `benchmark/tests/`

| Layer | Count (2026-05-23) | Scope |
|---|---|---|
| `tests/` | ~50 (5 files) | `doe_testbed` integration: case registry consistency, metadata, physics audit. |
| `benchmark/tests/` | 281 (19 files) | Method adapters, runner, statistics, paper-pipeline contracts. |

See [`tests/README.md`](../../tests/README.md) and the 19 `test_*.py` files
under `benchmark/tests/`.

### 2.7 Output layer — `benchmark/results/`

Generated content.  Never edited by hand; gitignored except for
small reference reports kept under `_archive/` for provenance.

```
benchmark/results/
├── full_benchmark/                  Stage 2 sweep output
│   ├── results.csv                  the headline data, 81 600 rows on a full run
│   ├── REPORT.md                    Stage 3 narrative
│   ├── figs/                        Stage 3 PNGs
│   ├── RUN_MANIFEST.json            Stage 10 archive
│   └── logs/                        per-cell heartbeat + warnings
├── ranking_reversal/                Stage 4 D1
│   ├── correlation.json             ρ + τ + CI for each (pairing × budget)
│   ├── pairing_dimension/
│   ├── pairing_topology/
│   ├── pairing_random/
│   └── analytical_results.csv       analytical phase (shared across pairings)
├── sample_complexity_0.20/          Stage 5 D2
│   ├── learning_curves.csv
│   ├── rule_summary.csv             with n_cases / exploratory columns
│   └── REPORT.md
├── reliability_recovery/            Stage 6 D3
│   ├── reliability_per_method.csv   with p_holm column
│   ├── reliability_long.csv
│   └── REPORT.md
└── _archive/                        legacy / pilot runs
```

## 3. Data flow

A single full publication run touches the following artefacts in order:

```
[Stage 0]  cantilever_beam_5d × default_pool × {5,10} × 2 reps
              → benchmark/results/_smoke/results.csv

[Stage 1]  validate_run_config.py    (read-only against preregistration.md)

[Stage 2]  benchmark/scripts/run_full_benchmark.py
              read:  doe_testbed registry
              write: benchmark/results/full_benchmark/results.csv
                     (streamed row-by-row, append-only)

[Stage 3]  benchmark/scripts/generate_report.py
              read:  results.csv
              write: REPORT.md, figs/*.png

[Stage 4]  benchmark/scripts/run_ranking_reversal.py --pairing-mode all
              read:  doe_testbed
              write: ranking_reversal/{analytical_results,physical_results}.csv,
                     ranking_reversal/correlation.json,
                     ranking_reversal/pairing_*/rank_comparison.csv
           benchmark/scripts/recompute_spearman_ci.py
              read:  ranking_reversal/pairing_*/rank_comparison.csv
              write: augments correlation.json with bootstrap CIs

[Stage 5]  benchmark/scripts/run_sample_complexity.py results.csv
              write: sample_complexity_0.20/rule_summary.csv,
                     learning_curves.csv

[Stage 6]  benchmark/scripts/run_reliability_recovery.py results.csv
              write: reliability_recovery/reliability_per_method.csv

[Stage 7]  optional A1/A2 sweeps (post-registration)

[Stage 8]  scripts/prepare_figdata.py
              read:  results.csv + ranking_reversal/ + sample_complexity_0.20/
                     + reliability_recovery/
              write: paper/figdata/fig{1..6}_*.csv

[Stage 9]  scripts/plots_nature.py
              read:  paper/figdata/
              write: paper/figures/{pdf,png}/fig{1..6}.{pdf,png},
                     paper/figure_manifest.json

[Stage 10] scripts/fill_paper_numbers.py
              read:  results.csv + ranking_reversal + sample_complexity +
                     reliability_recovery + figdata
              write: paper/numbers.{tex,json}
           scripts/generate_run_manifest.py
              read:  full_benchmark/
              write: full_benchmark/RUN_MANIFEST.json
```

Every step is **idempotent** and **resumable**; rerunning with the
same arguments and the same `results.csv` produces byte-identical
outputs (modulo wall-clock columns).

## 4. Editing rules of thumb

| If you want to … | Edit this | Don't touch |
|---|---|---|
| Add a new test case | `doe_testbed/cases/` + `doe_testbed/registry/` | `benchmark/`, `paper/preregistration.md` (post-freeze cases need an amendment) |
| Add a new DOE method | `benchmark/methods.py` (factory) + adapter file | `nature_pool()` is frozen; new methods go in `sota_pool()` |
| Tune a hyperparameter | `benchmark/hyperparameters.py` AND the factory in `benchmark/methods.py` | Inconsistency hard-blocks `validate_run_config.py` |
| Change a Discovery's decision rule | `paper/preregistration.md` only via `paper/preregistration_amendments.md` | Code changes alone do not move the protocol |
| Add a paper figure | `scripts/prepare_figdata.py` + `scripts/plots_nature.py` | `benchmark/scripts/generate_report.py` (REPORT.md only) |
| Bump a paper-text macro | `scripts/fill_paper_numbers.py` (single source of truth) | Never hand-edit `paper/numbers.tex` |

## 5. Cross-references

- [`README.md`](../../../README.md) — project overview (English)
- [Repository guide](../../../README.md) — project overview (中文)
- [`docs/README.md`](../README.md) — documentation directory index
- [`scripts/README.md`](../../scripts/README.md) — scripts catalogue + one-click pipeline launcher
- [`benchmark/README.md`](../../benchmark/README.md) — benchmark-suite architecture (per-module)
- [Repository guide](../../../README.md) — pre-registration + submission templates
- [`tests/README.md`](../../tests/README.md) — test layout
- [`doe_testbed/README.md`](../../doe_testbed/README.md) — case-library structure
- [`CHANGELOG.md`](../project/CHANGELOG.md) — version history
