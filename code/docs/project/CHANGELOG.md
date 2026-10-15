# Changelog

All notable changes to DOE-Benchmark are recorded here.  The current
canonical submission tree is `C:\wbl\DOE`; references to
`C:\DOE-benchmark` are retained only where needed for historical provenance.

The format follows the *Keep a Changelog* convention; versions follow
PEP 440 (`0.4.0.dev0` style for the package, `0.4.0-prereg` for the
pre-registration protocol version label).

> **Reading order**: top of the file is most recent.  Each entry lists
> the **scientific impact** explicitly so a reviewer can decide
> whether to re-read the relevant SAP / preregistration section.

---

## [Unreleased]

### Documentation

- Added top-level `ARCHITECTURE.md` with the 5-second view, layer-by-layer
  walkthrough, and Stage 0–10 data-flow diagram.
- Added `docs/GLOSSARY.md` — six-section terminology reference for
  project / DOE / statistics / engineering / code vocabulary.
- Added `docs/FAQ.md` — reproducibility + reviewer Q&A keyed to the
  pre-registration sections.
- Added `tests/README.md` and `doe_testbed/README.md` so each subdirectory
  has a navigation entry-point.
- Added this `CHANGELOG.md` (mirror-scoped).
- Refreshed `benchmark/README.md` test count (was 112, now 281).

---

## [0.4.0.dev0] — 2026-05-23 (audit + bug-fix pass)

### Fixed (scientific correctness)

- **Bug #1** — `benchmark/scripts/recompute_spearman_ci.py` silently
  skipped CI computation for the `topology` and `random` pairings when
  a legacy v0.3-style top-level `rank_comparison.csv` (no `pairing_mode`
  column) coexisted with the new `pairing_*/rank_comparison.csv`
  files.  Inverted the file-discovery preference so the nested layout
  always wins; legacy file is now a fallback only.
  *Scientific impact*: D1 §A3-finding (pairing-mode robustness)
  could not be computed under the previous code on this exact mirror
  layout — `correlation.json` topology + random entries were left
  without CI fields.  Pinned by
  `benchmark/tests/test_paper_pipeline.py::TestRecomputeSpearmanCi`
  (3 tests).

- **Bug #2** — `benchmark/reliability.py::reliability_metrics` returned
  `f_bal_acc_o{j} = 0.5 * (0 + tnr)` when the test set had zero true
  failures (`tp + fn == 0`), systematically underrating well-calibrated
  methods on edge-case test sets.  Now emits NaN when either truth
  class is empty, and the global `f_bal_acc_mean` aggregator skips
  non-finite per-output entries.
  *Scientific impact*: D3's `reliability_ranking(metric="f_bal_acc")`
  results were biased low on degenerate test sets.  Pinned by
  `benchmark/tests/test_paper_pipeline.py::TestReliabilityBalAccDegenerate`
  (4 tests).

- **Bug #3** — `scripts/prepare_figdata.py::make_fig4` labelled NaN
  signed-bias methods as `"none"` (the same label used for genuinely
  zero-bias methods).  Added a fourth `"undefined"` label for NaN
  inputs so §D3 paper text does not over-state the calibrated-method
  count.
  *Scientific impact*: §D3 paper-text conclusions on number of
  calibrated methods.  Pinned by
  `benchmark/tests/test_paper_pipeline.py::TestPrepareFigdata::test_fig4_bias_direction_distinguishes_nan_from_zero`.

### Fixed (documentation)

- Cleared 21 dead links pointing to legacy documentation paths from
  `README.md` and `README_zh.md`; replaced with local `docs/`,
  `paper/`, and root-level references.

- Cleared 5 dead links in `scripts/README.md`; replaced with the actual
  scripts present in this repository (`validate_run_config.py`, `audit_macros.py`,
  `check_submission_ready.py`, `prepare_figdata.py`,
  `plots_nature.py`, `fill_paper_numbers.py`,
  `generate_run_manifest.py`, `generate_si.py`).

- Added a `b'r"..."'` raw-string prefix to the module docstring of
  `benchmark/reliability.py` to silence the SyntaxWarning on the
  `\hat{P_f}` LaTeX in the docstring.

### Added

- **One-click pipeline orchestrator**:
  - `scripts/run_pipeline.py` — Python orchestrator wiring Stage 0–10
    with stage gating, output verification, dry-run mode, and per-stage
    profile-based defaults (`--profile smoke` vs `--profile full`).
  - `scripts/run_pipeline.ps1` — Windows PowerShell wrapper that pins
    BLAS thread caps before forwarding to the Python orchestrator.
  - `scripts/run_pipeline.sh` — Linux / macOS / WSL bash wrapper.
  - `benchmark/tests/test_pipeline_orchestrator.py` — 27 pytest cases
    covering stage table integrity, profile defaults, CLI selection,
    stage command builders, dry-run integration, BLAS pinning.

- **Doc-link integrity guard**:
  - `scripts/check_doc_links.py` rewritten to walk every `*.md` in the
    repo, supporting inline-style and reference-style links (the
    canonical Markdown link syntaxes), with anchor-fragment +
    external-URL handling.
  - `benchmark/tests/test_doc_links.py` — 7 pytest cases that fail CI
    on broken relative links *and* on accidental deletion of
    publication-track documents.

- **Color-blind-safe palette (GAP-F4)**:
  - `scripts/plots_nature.py` switched from `plt.cm.tab20` (Fig 3) and
    `plt.cm.RdYlBu_r` (Fig 5) to Wong's 8-colour CVD-safe palette and
    the `cividis` colormap.  Added `cvd_categorical_colors(n)` helper
    with Wong → cividis fallback for n > 8.
  - 5 pytest cases in
    `benchmark/tests/test_paper_pipeline.py::TestPlotsNatureColorblindPalette`
    pin the palette and grep-guard against re-introduction of unsafe
    colormaps.

- **Documentation**:
  - `docs/GAP_ANALYSIS_TO_NATURE_SUB.md` — distance to Nature
    sub-journal submission, P0/P1/P2 prioritised.
  - `docs/PROJECT_STATUS.md` — Stage 0–10 status board with real
    numbers from the on-disk `correlation.json`.
  - `docs/STATISTICAL_ANALYSIS_PLAN.md` — CONSORT-style SAP unifying
    pre-registration §1.x with the implementation contract.
  - `docs/REPRODUCIBILITY.md` — Stage 0–10 reproduction walkthrough.
  - `docs/README.md` — directory index for the four publication-track
    docs.

- **Paper templates**:
  - `paper/preregistration_decisions.template.md` — H1/H2/H3 decision
    template; SAP §10 calls for this file to be filled in *before* any
    manuscript-text edit.
  - `paper/significance_statement.template.md` — 120/150/200-word
    Editorial Summary template with per-journal trim guide.
  - `paper/cover_letter.template.md` — ≤ 1-page sub-journal cover
    template with NComms / NCS / NMethods / NMI customisation table.
  - `paper/preregistration.md` and `paper/preregistration_amendments.md`
    keep the project self-contained for the scientific protocol.
  - `paper/README.md` — explains the difference between locked
    (preregistration / amendments) and template files.

- **Module docstring** added to `scripts/generate_run_manifest.py`
  documenting the three-layer (code / machine / experiment) provenance
  capture and the script's read-only contract.

### Changed

- `scripts/README.md` re-organised into four categories (A: routine
  validation, B: figure / artefact generation, C: one-shot audits,
  D: sweep launchers), reflecting actual repository contents.

- `README.md` and `README_zh.md` gained a "Project documentation"
  section pointing to the root-level, `docs/`, and `paper/` trees.

- `paper/README.md` split files into "locked / append-only" vs
  "templates" tables.

### Verified

- 281 tests pass (`python -m pytest benchmark/tests`).
- 0 broken Markdown links across 19 files / 97 relative links.
- `scripts/run_pipeline.py --dry-run` exits 0 on both `smoke` and
  `full` profiles.

---

## [v0.4.0-prereg] — 2026-05-01 (frozen pre-registration protocol version)

This is the **scientific freeze**: case library, method pool, and
analysis plan locked before the 30-repeat headline sweep ran.  See
[`paper/preregistration.md`](../../../manuscripts/protocol/preregistration.md) for the
complete protocol.

### Frozen

- 34 cases (`doe_testbed/registry/`).
- 20 methods in `nature_pool()` (`benchmark/methods.py`).
- 4 budgets `{5, 10, 20, 50}·d`.
- 30 repeats per cell.
- 1 000-point Sobol' reference test set per case.
- `seed_strategy = "case_repeat"`, `seed_start = 1000`.
- Decision rules and falsification clauses for D1 / D2 / D3.

### Added at freeze

- `paper/preregistration.md` (locked).
- `paper/preregistration_amendments.md` (append-only; empty at freeze).
- `scripts/validate_run_config.py` enforcing frozen knobs at every
  launch.

---

## [Initial commit] — pre-2026-05-01

Forked from the doe-testbed v0.4.0-dev repository.  Earlier history is
kept in the upstream project records; this changelog tracks the current
DOE-Benchmark repository.

The fork inherited:

- 5 samplers (`Random`, `LHS`, `OptimisedLHS`, `Sobol`, `Halton`).
- 11 surrogates (`SURROGATE_REGISTRY`).
- 3 sequential active learners (EGO/EI, Max-Variance, AK-MCS).
- 2 BoTorch BO methods (qEI, SAASBO).
- `BenchmarkRunner` with paired-sample seed derivation, soft timeouts,
  resumable streaming, heartbeat logging.
- Friedman / Nemenyi / Holm / bootstrap CI / Cliff's δ statistical
  layer.
- Web FastAPI + Vue 3 SPA.
- 851 + 112 = 963 total tests across `tests/` and `benchmark/tests/`.
