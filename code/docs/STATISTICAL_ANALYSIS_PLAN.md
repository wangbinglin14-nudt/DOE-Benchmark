# Statistical Analysis Plan (SAP)

> **Purpose**.  This document is the single, machine-comparable SAP for
> the DOE-Benchmark v0.4.0 Nature-grade study.  It compiles the
> pre-registered hypotheses from
> [`../paper/preregistration.md`](../../manuscripts/protocol/preregistration.md), the
> three amendments in
> [`../paper/preregistration_amendments.md`](../../manuscripts/protocol/preregistration_amendments.md),
> and the implementation contract enforced by the test suite (`benchmark/tests/test_paper_pipeline.py`,
> `benchmark/tests/test_pipeline_orchestrator.py`,
> `benchmark/tests/test_prereg_compliance.py`) into a single
> CONSORT-style analysis plan.
>
> **Format**.  Follows the *Nature Portfolio* recommendation for
> registered analyses: H₀ / H₁, primary endpoint, decision rule,
> falsification clause, sensitivity analyses.
>
> **Locked since**.  v0.4.0-prereg (2026-05-01).  Freeze
> provenance is documented in `paper/preregistration.md`: the
> label is a version identifier, not a resolvable git tag, and
> the freeze date is an author attestation.
>
> **Audit gate**.  CI fails if `audit_macros.py` or `test_paper_pipeline.py`
> diverges from this document.  Any change here that is not also
> reflected in code + tests is illegal post-freeze.

---

## 1. Population, exposures and outcome (PICO-style framing)

| Element | Specification |
|---|---|
| **Population** | All cell-tuples `(case_id, method, budget_mult, repeat)` produced by `BenchmarkRunner` on the frozen `nature_pool()` (20 methods) over the 34 cases of `doe_testbed`. |
| **Exposures** | Method (20-level categorical), budget multiplier *m* ∈ {5, 10, 20, 50}, response-surface topology (7-level categorical, multi-label), failure-threshold present (binary, 16 of 34 cases). |
| **Outcomes** | Primary: `nrmse` (continuous). Secondary: `pf_pred_o*`, `pf_true_o*`, `f_bal_acc_o*`, `f_brier_o*`, `t_total_s`, `status`. |
| **Inclusions** | Cells with `status == "ok"` for inferential tests. Cells with `status ∈ {"failed", "timeout"}` retained in the raw CSV and counted in *failure-rate* tables but excluded from `nrmse` rank tests. |
| **Sample size** | 34 × 20 × 4 × 30 = **81 600 cells** (target). |

---

## 2. Pre-registered hypotheses

### 2.1 H1 — Discovery 1 (rank correspondence across task families)

| Item | Value |
|---|---|
| **Question** | Do method rankings on six analytical functions transfer to the physics-grounded subset under the protocol? |
| **H₀** | At budget = 10·d, Spearman ρ between the analytical-rank vector and the physical-rank vector ≥ 0.70. |
| **H₁** | Spearman ρ < 0.70. |
| **Test statistic** | Spearman ρ on N=20 method ranks, augmented by 95 % percentile bootstrap CI from 10 000 resamples of the *method axis*.  Earlier diagnostic runs may expose an 18-method non-BoTorch fallback, but the formal publication pipeline now requires BoTorch and blocks that fallback for the headline sweep. |
| **Decision rule (primary)** | Reject H₀ iff the **upper** 95 % CI bound of ρ at *m=10* is below 0.70. (Stricter than Fisher-z p-value alone.) |
| **Decision rule (secondary)** | Kendall τ with matching CI; sign and order must agree with Spearman ρ for the conclusion to be reported. |
| **Multiplicity adjustment** | None inside D1. D1 contributes one p-value to the family-wise Holm correction across {D1, D2, D3} (k=3, see §3). |
| **Robustness** | Repeat the analysis under *three pairing modes*: `dimension` (default), `topology` (Jaccard-maximal on `response_topology`), `random` (null distribution).  D1 is reported as *robust to pairing* iff ρ < 0.70 holds under both `dimension` and `topology` (amendment §A3 falsification). |
| **Falsification (§6 of preregistration)** | D1 is falsified iff the upper 95 % CI of ρ at 10·d ≥ 0.70 **and** both per-method shifts (random+gp, lhs_opt+rbf) have CIs including zero. |

### 2.2 H2 — Discovery 2 (sample-complexity rule of thumb)

| Item | Value |
|---|---|
| **Question** | Does the textbook *n = 10·d* rule (Loeppky, Sacks & Welch, 2009) hold across response-surface topologies on physical engineering cases? |
| **H₀** | ≥ 4 of the 7 topology classes pass at *m=10* (i.e. ≥ 50 % of methods reach NRMSE ≤ 0.20 in median over repeats). |
| **H₁** | ≤ 3 of 7 classes pass. |
| **Test statistic** | One-sided Clopper–Pearson upper bound on `pass_count / 7`. |
| **Decision rule (primary)** | Reject H₀ iff the 95 % one-sided CI upper bound of *p̂ = pass_count / 7* is below 4/7 ≈ 0.571. |
| **Threshold** | NRMSE ≤ 0.20 (practitioner target).  NRMSE ≤ 0.05 is reported in parallel as an extreme-target lower bound but not used for the headline. |
| **Multiplicity** | One p-value into Holm. |
| **Exploratory bucket** | Topologies with `n_cases < 3` are flagged `exploratory=True` in `rule_summary.csv` and **not counted** in the headline numerator/denominator.  The exploratory rows are reported separately in SI Table E.5. |
| **Falsification** | D2 is falsified iff ≥ 4 topology classes pass at *m=10*, NRMSE ≤ 0.20. The confirmatory test itself uses the one-sided Clopper–Pearson upper bound and rejects H₀ only when that bound is below 4/7. |

### 2.3 H3 — Discovery 3 (signed reliability bias)

| Item | Value |
|---|---|
| **Question** | On cases with a declared failure threshold, does the surrogate-predicted failure probability deviate from ground truth in a directional way? |
| **H₀** | The signed-bias distribution is centred at zero (operationally, median(*b*) = 0 for the signed-rank test), where *b* = P̂_f^surr − P̂_f^true. |
| **H₁ (primary, one-sided)** | median(*b*) > 0 (over-estimate / *conservative*). |
| **H₁ (secondary, two-sided)** | median(*b*) ≠ 0. |
| **Test statistic** | Wilcoxon signed-rank test on per-cell *b* values for the location hypothesis (16 cases × 30 repeats × budget = 480 cells / budget). |
| **Effect size and confidence intervals** | The pooled mean signed bias is the effect-size estimate; its 95 % percentile bootstrap CI uses **cluster-aware resampling** at the case level (16 clusters; primary), with an i.i.d. bootstrap reported as sensitivity. |
| **Decision rule** | Reject H₀ iff the H₁-primary p-value survives Holm correction at family-wise α=0.05 across {D1, D2, D3} **and** the cluster-aware CI for the mean effect excludes zero. This combines location evidence with precision for the reported mean. |
| **Aggregation** | For cases with multiple thresholded outputs, *output_aggregate=max* (the worst-failing output dominates).  Reported in §M2 of the manuscript with sensitivity to *mean* aggregation in Extended Data. |
| **Falsification (secondary check)** | The secondary two-sided Wilcoxon check has p > 0.05/3 **or** the bootstrap CI includes zero. The primary D3 decision remains the positive one-sided p-value with method-level Holm adjustment plus the case-cluster CI for the mean. |

---

## 3. Family-wise error control across the three Discoveries

We test three pre-registered hypotheses at family-wise α = 0.05 using
**Holm–Bonferroni** (Holm 1979).  Let p₍₁₎ ≤ p₍₂₎ ≤ p₍₃₎ be the
ordered p-values.  Reject H₍ⱼ₎ iff p₍ⱼ₎ ≤ α / (k − j + 1) for k=3:

| Rank | Threshold |
|---|---|
| smallest p | 0.05 / 3 = **0.0167** |
| middle p   | 0.05 / 2 = **0.025**  |
| largest p  | 0.05 / 1 = **0.05**   |

A Discovery is reported as **confirmed** iff:

1. its p-value is below the Holm threshold above, **and**
2. the corresponding effect-size CI excludes the null value.

> **Implementation invariant**: the function `_holm_adjust(pvals)` is
> implemented in `benchmark/reliability_recovery.py`, mirrored without
> drift in `scripts/prepare_figdata.py` and `scripts/fill_paper_numbers.py`.
> `benchmark/tests/test_paper_pipeline.py::TestHolmConsistency` pins the
> three implementations to identical numerical output.

---

## 4. Effect-size estimators and CIs

| Discovery | Effect size | CI estimator | n_boot | Cluster |
|---|---|---|---:|---|
| D1 | Spearman ρ; Kendall τ | percentile bootstrap, resample method axis | 10 000 | none |
| D2 | proportion of topology classes passing | Clopper–Pearson one-sided | exact | none |
| D3 | mean signed bias | percentile bootstrap, **cluster** at case_id | 10 000 | case_id (n=16) |

The cluster-aware bootstrap in D3 is the *primary* per pre-registration
§5.  An i.i.d. bootstrap is computed in parallel; if cluster-aware and
naïve CIs disagree on sign or significance, both are reported with
cluster-aware as the headline.

---

## 5. Pre-specified sensitivity analyses

These are **additional** analyses, run regardless of the headline outcome,
to bound the robustness of each Discovery.

### 5.1 D1 sensitivity

| ID | Sensitivity | Pre-specified outcome | Where reported |
|---|---|---|---|
| D1-S1 | Spearman ρ at 20·d in addition to 10·d | numerical | §D1 + Fig 2 |
| D1-S2 | Three pairing modes (`dimension`/`topology`/`random`) | concordance test (§A3-finding) | §D1 Robustness paragraph + SI |
| D1-S3 | Per-method rank-shift CI | ≥ 1 method with CI excluding 0 → reported | §D1 |

### 5.2 D2 sensitivity

| ID | Sensitivity | Where |
|---|---|---|
| D2-S1 | NRMSE ≤ 0.05 (extreme target) | parallel column in `rule_summary.csv` |
| D2-S2 | Per-method (not per-topology) `n*/d` learning curves | SI Fig E.4 |
| D2-S3 | Failure-rate retained in denominator (penalised D2) | SI Table E.5 |

### 5.3 D3 sensitivity

| ID | Sensitivity | Where |
|---|---|---|
| D3-S1 | i.i.d. vs cluster-by-case bootstrap CI width ratio | SI Table E.3 |
| D3-S2 | `output_aggregate=mean` vs `=max` | Extended Data Fig E.3 |
| D3-S3 | Method-level Holm vs BH-FDR | SI §si_multiple_testing.tex |

---

## 6. Subgroup / stratified analyses

The following **descriptive** breakdowns accompany every Discovery; they
are *not* hypothesis-tested and carry no Holm correction:

- Stratification by **`response_topology`** (D2 is by definition stratified; D1/D3 carry it as descriptive).
- Stratification by **input dimension bucket** (`d ≤ 5`, `5 < d ≤ 10`, `d > 10`).
- Stratification by **engineering domain** (structures / heat / aero / chemical / multidisciplinary / geomechanics).
- Stratification by **`cost_level`** (`L0` / `L1+`).

Each stratified analysis is in `benchmark/stratified.py::stratified_ranks`
and surfaces in the Stage 8 figdata CSVs (`fig5_mechanism.csv`).

---

## 7. Power considerations (a-priori)

| Test | k | n_eff | Smallest detectable effect (α=0.05, power=0.8) | Source |
|---|---:|---:|---|---|
| D1 Spearman ρ < 0.70 | 1 | 20 (methods) | ρ_true ≤ 0.55 → upper CI < 0.70 with prob ≥ 0.8 | bootstrap simulation (§benchmark/tests) |
| D2 ≤ 3 / 7 pass | 1 | 7 | true pass-rate ≤ 0.12 → CP upper-bound < 4/7 with prob ≥ 0.8 | Clopper–Pearson exact (recomputed 2026-09-14) |
| D3 Wilcoxon one-sided | 1 | 480 cells × 30 reps cluster-corrected to ~16 indep blocks | abs effect ≥ 0.04 (Cohen's d ≈ 0.5) | Wilcoxon sample-size formula (Lehmann 1975) |

> **D2 caveat**.  N=7 topology classes is fixed by the case library, so the
> minimum detectable effect is intrinsically large.  An earlier version of
> this row stated that a true pass-rate of 0.40 gave power ≥ 0.8; that was
> wrong.  Under the registered rule the one-sided Clopper–Pearson upper bound
> must fall below 4/7, which happens only when at most one of the seven
> classes passes (k = 0: 0.348; k = 1: 0.521; k = 2: 0.659).  At the observed
> k = 3 the bound is 0.775, so rejection is arithmetically impossible and the
> test is direction-limited as well as low-powered.  Achieved power at a true
> pass-rate of 0.40 is 0.159, and power 0.8 requires a true pass-rate ≤ 0.12.
> This is reported as a known limitation in §M2 *Statistical caveats*, and the
> manuscript reports the D2 outcome as a point estimate rather than as a
> confirmed test.

---

## 8. Multiple-comparison guard for figure-level tests

Beyond the three D-level p-values, the manuscript reports *per-method*
or *per-topology* breakdowns.  These are controlled differently:

| Sub-table | n_tests | Correction | Reported as |
|---|---:|---|---|
| D3 per-method signed bias | 20 (methods) | Holm at 0.05 within D3 | `p_holm` column in `reliability_per_method.csv` |
| D2 per-topology learning-curve significance | 7 (topologies) | none — descriptive only | `learning_curves.csv` |
| Friedman post-hoc on full ranking (Stage 3) | 20 × 19 / 2 = 190 pairs | Nemenyi critical-difference | CD diagram in `REPORT.md` |
| Stratified Friedman per topology / domain | Friedman p per stratum | none, descriptive | `stratified_ranks.csv` (figdata) |

---

## 9. Pre-specified analysis pipeline (executable)

The full SAP is operationalised as Stage 0 → 10 of
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).  In short:

```bash
# Formal server launch path
python scripts/check_server_launch_ready.py
bash scripts/run_pipeline.sh

# Stage 1 — pre-flight gate run by the pipeline
python scripts/validate_run_config.py \
    --pool nature --budgets 5 10 20 50 \
    --n-repeats 30 --n-test 1000 \
    --seed-strategy case_repeat --seed-start 1000 --cases ALL

# Stage 2 — owned by the pipeline.  The low-level command below is only for a
# reviewed Stage 2 resume/diagnostic; it does not create the complete paper
# archive by itself.
python benchmark/scripts/run_full_benchmark.py \
    --cases ALL --pool nature --budgets 5 10 20 50 --n-repeats 30 \
    --seed-strategy case_repeat --seed-start 1000 \
    --confirm-large-run --require-botorch --require-formal-case-backends --require-runtime-monitoring \
    --output-dir benchmark/results/full_benchmark

# Stage 4 — D1 (3 pairings + bootstrap CI)
python benchmark/scripts/run_ranking_reversal.py \
    --pool nature --pairing-mode all --n-repeats 30 \
    --output-dir benchmark/results/ranking_reversal
python -m benchmark.scripts.recompute_spearman_ci \
    --rr-dir benchmark/results/ranking_reversal --n-boot 10000

# Stage 5 — D2
python benchmark/scripts/run_sample_complexity.py \
    benchmark/results/full_benchmark/results.csv \
    --threshold 0.20 --rule-budget 10 \
    --output-dir benchmark/results/sample_complexity_0.20

# Stage 6 — D3 (cluster-aware as primary)
python benchmark/scripts/run_reliability_recovery.py \
    benchmark/results/full_benchmark/results.csv \
    --budget-mult 10 --cluster-by case_id \
    --output-dir benchmark/results/reliability_recovery

# Stage 8-10 — figdata + figures + numbers + manifest after Stage 2 exists
python scripts/run_pipeline.py --from-stage 8 --confirm-full-run
```

A single `bash scripts/run_pipeline.sh` on the designated Linux server is the
formal way to execute Stage 2 through Stage 10 without splitting this contract.

---

## 10. Decision document (post-sweep)

After Stage 2 finishes, the assistant **must** write a one-page
`paper/preregistration_decisions.md` *before* any manuscript text edit,
following this template:

```markdown
# Pre-registration decisions — DOE-Benchmark v0.4.0

Run manifest: `benchmark/results/full_benchmark/RUN_MANIFEST.json`
Sweep finished: <UTC ISO-8601>

## D1 — H1: ρ < 0.70 at 10·d

- Observed ρ (dimension)  = X.XXX [CI XX, XX]
- Observed ρ (topology)   = X.XXX [CI XX, XX]
- Observed ρ (random)     = X.XXX [CI XX, XX]
- Holm-adjusted p         = ...
- Decision: **{Confirmed | Falsified | Inconclusive}**
- Manuscript headline change required: <yes/no, brief description>

## D2 — H2: ≤ 3 / 7 topology classes pass

- pass_count = N / 7  (CP 95 % CI [LO, HI])
- Holm-adjusted p = ...
- Exploratory topologies excluded (n_cases < 3): {list}
- Decision: **{Confirmed | Falsified}**

## D3 — H3: positive signed-bias location (median(b) > 0)

- Mean signed bias    = X.XXX
- Median signed bias  = X.XXX (location statistic for Wilcoxon)
- Cluster-aware CI    = [LO, HI]
- i.i.d. CI           = [LO, HI]
- Wilcoxon one-sided p (>) = ...
- Holm-adjusted p     = ...
- Direction           = {over | under | mixed | none}
- Decision: **{Confirmed | Falsified}**

## Falsification log

For each Discovery whose decision is **Falsified** or **Inconclusive**,
record the §6-clause that triggered, plus the exact rewording required
in `nature_cs_main.tex` §D1/§D2/§D3.
```

This document is the only legal route from sweep results to manuscript
text changes.

---

## 11. Departures from the SAP

Any departure (e.g. a methodological correction discovered post-freeze)
must:

1. Be appended to `paper/preregistration_amendments.md` with a new
   amendment ID (`A4`, `A5`, ...).
2. Trigger a new commit and, once the archive is versioned under
   public version control, a new protocol version label.
3. Re-run `scripts/check_submission_ready.py`.
4. Be cross-referenced from the manuscript §Methods (so reviewers can
   audit *exactly* what changed).

The §A1 / §A2 / §A3 amendments to date are listed in
[`../paper/preregistration_amendments.md`](../../manuscripts/protocol/preregistration_amendments.md).
