# FAQ

> **Audience.**  Two groups: (a) someone who wants to **reproduce** the
> published results (replicators), and (b) someone reading the
> manuscript and wondering whether a methodological choice was
> defensible (reviewers).
>
> Each Q is followed by the file path or doc that contains the
> definitive answer, so this FAQ stays out of the way of the canonical
> references.

## A. Big-picture

### A.1 What problem does DOE-Benchmark solve?

The DOE / surrogate-modelling literature has for four decades validated
methods on a small set of analytical functions (Branin, Hartmann,
Ishigami, ...).  Whether method rankings on those analytical functions
predict rankings on **physics-grounded engineering problems** has never
been measured at scale.  DOE-Benchmark provides 34 physics-grounded
test cases under a single `case.evaluate(X)` interface and benchmarks
20 DOE / surrogate / Bayesian-optimisation methods on them with a
pre-registered statistical protocol.  See [`README.md`](../../README.md)
for the project pitch and
[`paper/preregistration.md`](../../manuscripts/protocol/preregistration.md) §1 for the
formal scientific question.

### A.2 What is the current project root?

`C:\wbl\DOE` is the current canonical submission tree.  It contains the
case library, benchmark engine, tests, pre-registration documents, paper
templates, reproducibility docs, and pipeline scripts.  The older
`C:\DOE-benchmark` tree is historical and must not be used as the source of
the headline result.  See
[`PROJECT_STRUCTURE.md`](overview/PROJECT_STRUCTURE.md) for the directory map
and command entry points.

### A.3 What is "pre-registered"?

Three things were frozen on **2026-05-01** under protocol version
`v0.4.0-prereg`.  That label is a version identifier, not a
resolvable git tag: the freeze date is an author attestation,
because the public git history postdates it (see the
freeze-provenance note in `paper/preregistration.md`):

1. The case library (`doe_testbed/`, 34 cases).
2. The method pool (`benchmark.nature_pool()`, 20 methods).
3. The statistical analysis plan (H₀/H₁/decision rules per Discovery).

Any change after the freeze that affects scientific conclusions
requires an entry in
[`paper/preregistration_amendments.md`](../../manuscripts/protocol/preregistration_amendments.md).
Three amendments exist: A1 (SOTA methods), A2 (multi-fidelity), A3 (D1
pairing-mode robustness).  See
[`docs/STATISTICAL_ANALYSIS_PLAN.md`](STATISTICAL_ANALYSIS_PLAN.md) §11.

## B. Reproducibility

### B.1 What's the minimum hardware to reproduce?

| Profile | RAM | Cores | Wall clock | Use |
|---|---|---|---|---|
| `--profile smoke` | 4 GB | 1 | ≈ 5 min | Sanity check on a laptop |
| Full sweep on Linux | ≥ 64 GB | ≥ 32 | ≈ 21 h | Headline reproduction |
| Windows workstation | 16 GB | 8+ | n/a | Smoke, unit, physics, dry-run/preflight checks only |

See [`docs/REPRODUCIBILITY.md`](REPRODUCIBILITY.md) §0.  The full
34-case x 20-method x 4-budget x 30-repeat archive is a designated Linux
server job, not a local workstation job.  SAASBO at budget 50·d is the
memory-tightest cell; on the server, the formal pipeline sets heavy-method
parallelism conservatively.

### B.2 How do I reproduce just the figures from already-published data?

```
python scripts/run_pipeline.py --from-stage 8 --confirm-full-run
```

This skips Stages 0–7 (the sweep) and reruns Stages 8–10 (figdata,
figures, paper macros, manifest) from `benchmark/results/full_benchmark/results.csv`.
You need that CSV in place; you can fetch it from the Zenodo archive
that ships with the published paper.

### B.3 Is the reproduction bit-exact?

For numerical contracts on `nrmse`, `pf_pred_*`, `f_bal_acc_*`,
`spearman_rho`, `signed_bias_mean`, `p_holm` — **yes**, given the same
seeds and the same package versions pinned in
`requirements.lock`.  See [`docs/REPRODUCIBILITY.md`](REPRODUCIBILITY.md)
§14.  Wall-clock columns and worker scheduling order are non-deterministic
and explicitly excluded from the bit-exactness contract.

### B.4 Where are the seeds?

Per-cell seeds are derived from `(seed_start, case_id, repeat)` via
`blake2b(...).digest_size=4 & 0x7FFFFFFF` in `benchmark.core.derive_seed`.
The freeze value `seed_start = 1000` is in
[`paper/preregistration.md`](../../manuscripts/protocol/preregistration.md) §3 and
verified by `scripts/validate_run_config.py` before every launch.

### B.5 The sweep crashed mid-way.  What now?

On the server, re-run the formal pipeline command:

```
bash scripts/run_pipeline.sh
```

The orchestrator passes `--skip-existing` to Stage 2, and the resume gate
rejects corrupt or stale `results.csv` files before any cell is skipped.
Completed cells in a valid partial CSV are skipped and the sweep resumes from
the next pending cell.  The seed schedule is invariant under restart because
`derive_seed` is a pure function of `(seed_start, case_id, repeat)`.

### B.6 Why does the smoke test use `default_pool`, not `nature_pool`?

The smoke gate is a 5-minute sanity check on a laptop; loading the
20-method `nature_pool()` (which includes BoTorch SAASBO) requires
500 MB+ of imports and would dominate the smoke runtime.  The
10-method `default_pool` exercises every layer of the runner without
the heavy methods.  See `scripts/run_pipeline.py::SMOKE_POOL`.

## C. Scientific protocol

### C.1 Why 30 repeats per cell?

The minimum that gives a stable critical-difference at α=0.05 for
20 methods on 34 cases per Demšar (2006), and the practitioner
convention in the surrogate-modelling literature.  Pre-registered in
[`paper/preregistration.md`](../../manuscripts/protocol/preregistration.md) §3.

### C.2 Why these 34 cases and not others?

Each case had to satisfy *all* of: (i) governing equations published
with formula numbers; (ii) numerical solver implementable in pure
Python in O(seconds); (iii) a clearly-defined input domain; (iv) at
least one engineering output of practical interest.  See
[case overview](CASE_OVERVIEW.md) and the per-case
`docstring` of `doe_testbed/cases/*.py`.

### C.3 Why these 20 methods and not 50?

Each method had to satisfy: (i) at least one published paper /
textbook citation; (ii) implementable from public Python libraries;
(iii) reasonably bounded memory on a 16 GB host.  The `sota_pool()`
amendment (A1) adds TURBO + D-optimal post-registration to address
reviewer concerns about specific omissions.

### C.4 Why is `nature_pool` *frozen*?

Adding methods after seeing data turns confirmatory analysis into
exploratory analysis.  The frozen pool is what supports the headline
D1/D2/D3 conclusions; new methods can be added but only via the
*amendment* mechanism (`sota_pool`, `mf_pool`) and only reported in
**Extended Data**, not the headline.

### C.5 The headline budgets are {5, 10, 20, 50}·d.  Why not {1, 2, ..., 100}·d?

The four chosen levels span the practitioner-relevant range:
sample-starved (5·d), the textbook recommendation (10·d), comfortable
(20·d), and "as much data as we'd realistically allow" (50·d).  Using
fewer levels gives sharper Friedman p-values and avoids implicit
multiple-budget testing without correction.

### C.6 Why is the bootstrap CI on D3 a *cluster* bootstrap?

D3 has 16 cases × 30 repeats per case.  Repeats within a case share
the case's intrinsic difficulty (sometimes called the "case effect"),
so per-cell observations are *not* i.i.d.  A naïve bootstrap would
underestimate the standard error.  Cluster-resampling at the
`case_id` level (as the *primary* CI estimator per
[`paper/preregistration.md`](../../manuscripts/protocol/preregistration.md) §1.3) is the
textbook fix (Field & Welsh 2007).

### C.7 Why test ρ at *m=10* specifically?

Pre-registered as the primary endpoint because *m=10* is exactly the
budget targeted by the textbook *10·d* rule of thumb (Loeppky et
al. 2009) — that's the budget where method rankings *should* matter
most for the practitioner.  *m=20* is reported as the secondary
endpoint to show whether disagreement at *m=10* persists at higher
budgets.

### C.8 What happens to failed / timed-out cells?

`status ∈ {"failed", "timeout"}` rows are kept in the raw CSV but
**excluded** from inferential tests on `nrmse`.  They are **counted**
in the failure-rate denominator and the penalised Pareto analysis
(see `scripts/prepare_figdata.py::make_fig6` `analysis_type =
"penalized_failed_in_denominator"`).  D2 keeps method × topology cells
that fail-everywhere as `n_methods_never_pass`.  D3 method-cells with
`n_runs < 10` are flagged `low_n=True` and demoted to *exploratory*.

### C.9 What are the falsification clauses?

For each Discovery, the pre-registration §6 specifies the conditions
under which the headline must be rewritten as falsified rather than
confirmed.  Currently:

- **D1**: upper 95 % CI of ρ at 10·d is ≥ 0.70 *and* both per-method
  shifts (random+gp, lhs_opt+rbf) have CIs including zero.
- **D2**: ≥ 4 of 7 (non-exploratory) topology classes pass at *m=10*.
- **D3**: Wilcoxon two-sided p > 0.05 / 3 *or* the bootstrap CI
  includes zero.

### C.10 What if the data falsifies a Discovery?

`paper/preregistration_decisions.template.md`
must be filled in *before* any manuscript text edit, with the
`Decision = Falsified` field set and the pre-stated rewording quoted
verbatim.  The §A audit trail in `preregistration_amendments.md`
records the falsification.

## D. Statistical methodology

### D.1 Why Holm and not Bonferroni?

Holm has the same family-wise error guarantee as Bonferroni but is
strictly more powerful (rejects more true alternatives at the same
α).  Per Demšar (2006) recommendation for benchmark-method comparisons.
Implementation: `benchmark.reliability_recovery._holm_adjust()` (and
mirrored consistently across `prepare_figdata.py` /
`fill_paper_numbers.py`, pinned by
`benchmark/tests/test_paper_pipeline.py::TestHolmConsistency`).

### D.2 Why not BH-FDR?

BH controls the false-discovery rate, not the family-wise rate.  For a
3-Discovery family, FWER is the standard convention (Nosek et al. 2018,
Nature Portfolio guidance).  We additionally report Bonferroni and BH
as sensitivity in the Supplementary Information (source in
`paper/supplementary.tex`, table fragments in `paper/tables/`).

### D.3 Why bootstrap CI rather than Fisher-z CI on ρ?

The bootstrap CI is simulation-based and makes no parametric
assumption about the sampling distribution of ρ.  Fisher-z assumes
the rank correlation is approximately normal under the z-transform —
fine for ρ in [0.3, 0.7] but breaks down near boundaries (and our ρ
estimates are 0.88–0.97).  Pre-registered as percentile bootstrap with
n_boot = 10 000.

### D.4 The `f_bal_acc` metric reports NaN sometimes.  Why?

Balanced accuracy is undefined when the test set has zero true
positives or zero true negatives (no failure or all-failure).
Returning a numeric "0.5" would conflate "method got everything wrong"
with "test set is degenerate"; we emit NaN and downstream Friedman /
ranking tests drop those cells.  See `benchmark/reliability.py:115-127`
and the regression test
`benchmark/tests/test_paper_pipeline.py::TestReliabilityBalAccDegenerate`.

### D.5 Why "max" aggregation for multi-output failure bias?

A case with two thresholded outputs gets a per-output bias vector;
the "max-abs-bias" output is the *worst-failing* one and dominates a
conservative engineering decision.  Default aggregation per
[`paper/preregistration.md`](../../manuscripts/protocol/preregistration.md) §5; sensitivity
to the alternative `mean` aggregation is reported in Extended Data
Fig. E.3.

### D.6 Why not just use one Test set instead of bootstrapping?

The bootstrap is on the *ranks-across-methods* axis, not on the test
set.  The test set is fixed at 1 000 Sobol points per case.  The
bootstrap quantifies "how would the rank vector change if we'd
sampled a different set of 20 methods?".  Test-set MC noise is a
separate, smaller error component; reported in Extended Data §SI.4.

## E. Code, infrastructure, and policy

### E.1 Where do I file a bug?

Use the repository issue tracker.  Include the output of:

```
python scripts/run_pipeline.py --dry-run --profile smoke
```

so we can pin the environment + git commit.

### E.2 Can I add a new method?

Yes, but only into `sota_pool()` (or a new amendment-tagged pool).
Adding to `nature_pool()` after the freeze breaks pre-registration.
Steps:

1. Implement the adapter in `benchmark/methods.py` (or `bo_methods.py` for BO).
2. Register hyper-parameters in `benchmark/hyperparameters.py`.
3. Add a regression test in `benchmark/tests/test_new_methods.py`.
4. Add an amendment entry in `paper/preregistration_amendments.md`.

CI fails on hyperparameter drift, missing tests, or missing amendment.

### E.3 Why are there 40 ruff lint warnings?

Stylistic items only (import order, `dict()` vs `{}`, `zip()` without
`strict=`, unused defensive imports).  Not bugs.  The 2026-05-23 audit
catalogued each one; see git log entry "code review from-the-top".
We don't auto-fix because some are inside type-annotation blocks or
`if TYPE_CHECKING` guards that ruff misclassifies.

### E.4 Why does `correlation.json` sometimes lack `spearman_rho_ci_lo`?

Before the 2026-05-23 fix, when a directory contained both a legacy
top-level `rank_comparison.csv` (no `pairing_mode` column) and the new
`pairing_*/rank_comparison.csv` files, the `recompute_spearman_ci.py`
script preferred the legacy file and silently skipped CI computation
for `topology` and `random` pairings.  The fix prefers nested files
when any exist.  See `benchmark/tests/test_paper_pipeline.py::TestRecomputeSpearmanCi`
and the [`CHANGELOG.md`](project/CHANGELOG.md) entry for 2026-05-23.

### E.5 What's the difference between `_archive/` directories and live results?

`_archive/` directories under `benchmark/results/` and the canonical
clone retain pilot / legacy runs for provenance.  They are
*intentionally not aligned* with the v0.4 schema (e.g. they may use
the old 15-method extended pool); audit tools skip them by name.

## F. Submission

### F.1 What do I send to the journal?

The current files are being revised; no active submission package exists. At
submission, prepare and verify these materials from the live workspace:

1. The main PDF and editable source from `01_manuscripts/pdf/` and
   `01_manuscripts/nature_cs_main.tex`.
2. The six figures from `01_manuscripts/figures/` with their source data from
   `01_manuscripts/figdata/`.
3. The Supplementary Information source and PDF from `01_manuscripts/`, with
   the S1--S13 fragments from `01_manuscripts/tables/`.
4. The cover-letter draft from `01_manuscripts/editorial/cover_letter.md`.
5. The Editorial / Significance Statement from
   `01_manuscripts/editorial/significance_statement.md`.
6. The frozen protocol and amendments from `01_manuscripts/protocol/`.
7. The decision document `01_manuscripts/protocol/preregistration_decisions.md`
   (filled-in template) so reviewers see what we decided **before**
   editing the manuscript.
8. The complete versioned data and code deposit and its persistent DOI once
   the deposit actually exists.

The decision record makes the intended confirmatory and exploratory
boundaries visible. Its freeze date remains author-attested, so the record
does not itself establish an independent pre-sweep timestamp.

### F.2 What's the submission checklist?

Run:

```
python scripts/check_submission_ready.py paper/
python scripts/audit_macros.py
python scripts/check_doc_links.py
python -m pytest benchmark/tests
```

All four must exit 0 before you upload.  See
[`docs/REPRODUCIBILITY.md`](REPRODUCIBILITY.md) §13.

### F.3 Which sub-journal?

The current target is *Communications Engineering*. The working cover letter
is in `01_manuscripts/editorial/cover_letter.md`.
Earlier journal-target comparisons are historical and do not set the current
submission plan.

## G. Cross-references

- [`README.md`](../../README.md) — project overview
- [`ARCHITECTURE.md`](overview/ARCHITECTURE.md) — file-tree layout & data flow
- [`docs/GLOSSARY.md`](GLOSSARY.md) — terminology
- [`docs/STATISTICAL_ANALYSIS_PLAN.md`](STATISTICAL_ANALYSIS_PLAN.md) — formal SAP
- [`docs/REPRODUCIBILITY.md`](REPRODUCIBILITY.md) — Stage 0–10 walkthrough
- [`paper/preregistration.md`](../../manuscripts/protocol/preregistration.md) — frozen protocol
- [`paper/preregistration_amendments.md`](../../manuscripts/protocol/preregistration_amendments.md) — A1/A2/A3
- [`CHANGELOG.md`](project/CHANGELOG.md) — version history
