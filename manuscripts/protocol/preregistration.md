# Pre-registration — DOE-Benchmark Nature-grade sweep

> **Purpose.** This document freezes the scientific hypotheses,
> primary analyses, and decision rules of the DOE-Benchmark
> v0.4.0 Nature-grade study **before** the 30-repeat full sweep
> produces its results.  It follows the pre-registration guidance
> of Nosek et al. (2018, *PNAS*) and the Nature Portfolio
> *registered-analysis* conventions.

**Pre-registered by.**  Binglin Wang (corresponding author).
**Freeze provenance.**  The protocol version is labelled
`v0.4.0-prereg`, and that label is a version identifier rather than a
resolvable git tag: the public git history was created after the freeze,
so no commit or tag in it predates 2026-05-01 and the freeze date below
cannot be corroborated from version control.  The date is an author
attestation.  The frozen values themselves are independently verifiable,
because they are fixed by this document and enforced by
`benchmark/tests/test_prereg_compliance.py` and the method-source
fingerprints in `benchmark/results/full_benchmark/RUN_MANIFEST.json`.  A
third-party timestamped deposit of this file is a required submission
action; see `docs/SUBMISSION_EXTERNAL_INPUT_TEMPLATE_20260901.md`.  Any
modifications to this document after the freeze will be logged in
`paper/preregistration_amendments.md` with rationale and timestamp.
**Freeze date.**  2026-05-01 (UTC+08:00), prior to the 30-repeat
`benchmark/scripts/run_full_benchmark.py --pool nature
--n-repeats 30` execution.

---

## 1 · Motivation and hypotheses

The DOE / surrogate-modelling literature has for four decades
validated methods almost exclusively on a small set of analytical
test functions.  The **primary scientific question** of this study
is whether method rankings calibrated on analytical functions
generalise to physics-grounded engineering problems.

Three pre-registered hypotheses derive from that question.  The
hypotheses were identified during an exploratory pilot
(`benchmark/results/ranking_reversal/` v0.3, 6 repeats,
15 methods, 3 budgets) and are now **frozen** before the
confirmatory 30-repeat sweep.

### H1 (Discovery 1) — Rank correspondence across task families

> **H1** — At the practically-relevant training budget $n = 10\,d$,
> the Spearman rank correlation $\rho$ between the analytical-function
> rank vector and the physical-case rank vector is below the
> conventional equivalence threshold of $\rho = 0.70$.

- **Null ($H_0$)**: $\rho \ge 0.70$.
- **Alternative ($H_1$)**: $\rho < 0.70$.
- **Test statistic**: Spearman $\rho$ on the $N = 20$ method ranks,
  augmented by a percentile-bootstrap 95 % CI over the method axis
  (10 000 replicates).
- **Decision rule (primary)**: reject $H_0$ iff the **upper** 95 % CI
  bound of $\rho$ is below 0.70.  (This is stricter than the Fisher-$z$
  $p$-value alone; it requires the whole CI to lie under the threshold.)
- **Decision rule (sensitivity)**: also report Kendall $\tau$ with the
  matching bootstrap CI.  The two measures must agree in sign and
  order for the conclusion to be reported.

### H2 (Discovery 2) — Sample-complexity rule-of-thumb

> **H2** — When response-surface topology is treated as a
> stratification variable, the textbook "$n = 10\,d$ is sufficient"
> rule of thumb (Loeppky, Sacks & Welch, 2009) fails on a majority
> of the 7 topology classes in *DOE-Benchmark*.

- **Operationalisation of the rule**: a topology class *passes* at
  $m = 10$ if **a majority of the method pool** attains median
  NRMSE $\le 0.20$ over repeats.  The 0.20 threshold is the
  practitioner-friendly relaxation of Loeppky et al.'s implicit
  Gaussian-residual target ($\pm 10$ % of the range, equivalent to
  NRMSE $\le 0.10$).  The stricter 0.05 threshold is reported in
  parallel as an extreme-target lower bound.
- **Null ($H_0$)**: $\ge 4$ of 7 topology classes pass at $m = 10$,
  NRMSE $\le 0.20$.
- **Alternative ($H_1$)**: $\le 3$ of 7 topology classes pass.
- **Test statistic**: one-sided Clopper–Pearson **upper** confidence bound on
  $\hat p = \text{(pass count)} / 7$; reject $H_0$ iff this 95 % upper bound
  is below $4/7 = 0.571$.

### H3 (Discovery 3) — Reliability / failure-probability bias

> **H3** — On cases with a declared failure threshold, the signed-bias
> distribution $b = \hat P_f^{\text{surr}} - \hat P_f^{\text{true}}$ has a
> positive location (with a non-zero location as the two-sided alternative).
> The pooled **mean** signed bias is the reported effect-size estimate; the
> registered Wilcoxon test targets the distribution's location rather than
> claiming a direct test of the mean.

- **Null ($H_0$)**: the signed-bias distribution is centred at zero
  (operationally, $\operatorname{median}(b)=0$ for the signed-rank test).
- **Alternative (primary, one-sided)**: $\operatorname{median}(b)>0$.
- **Alternative (secondary, two-sided)**: $\operatorname{median}(b)\ne 0$.
- **Test statistic**: Wilcoxon signed-rank test on the per-cell $b$
  values for the location hypothesis; the pooled mean is accompanied by a
  95 % percentile-bootstrap CI with **cluster-aware resampling** at the case
  level (16 clusters) to respect within-case dependence.
- **Decision rule**: reject $H_0$ iff the $H_1^{\text{primary}}$
  $p$-value survives the Holm correction at family-wise $\alpha = 0.05$
  across the three Discoveries and the cluster-bootstrap CI for the mean
  excludes zero.  Thus the reported decision combines location evidence from
  Wilcoxon with an uncertainty interval for the mean effect size.
---

## 2 · Family-wise error control

All three hypotheses are tested at family-wise $\alpha = 0.05$ using
the Holm–Bonferroni procedure.  With $k = 3$:

| Rank in ordered $p$-values | Threshold |
|---|---|
| smallest | $\alpha_1 = 0.05 / 3 = 0.0167$ |
| middle   | $\alpha_2 = 0.05 / 2 = 0.025$  |
| largest  | $\alpha_3 = 0.05 / 1 = 0.05$   |

A Discovery is reported as *confirmed* iff its $p$-value is below the
Holm threshold **and** the corresponding effect-size CI excludes the
null value.

---

## 3 · Pre-registered benchmark protocol

- **Methods.**  `nature_pool()` with 20 methods: the frozen
  `full_method_pool()` plus the two BoTorch methods `botorch_qei` and
  `botorch_saasbo`.  Per amendment A0, the Nature-family headline
  archive requires the complete 20-method environment; a missing
  BoTorch stack is a launch blocker, not a permitted headline
  degradation.
- **Cases.**  All 34 cases of the *DOE-Benchmark* suite at the freeze
  commit.  Case identifiers, input ranges, topology tags, failure
  thresholds, and ground-truth Sobol indices are frozen in
  `doe_testbed/` and locked by the per-case metadata and bounds
  consistency checks in `tests/test_cases.py`.
- **Budgets.**  $n \in \{5d, 10d, 20d, 50d\}$, four levels.
- **Repeats.**  $n_{\text{repeats}} = 30$ per (case, method,
  budget) cell.
- **Seed derivation.**  `benchmark.core.derive_seed` with strategy
  `case_repeat`; each (case_id, repeat) tuple deterministically
  produces the seed used by *every* method in that cell.  This
  preserves the paired-sample structure required by Friedman and
  Wilcoxon signed-rank tests.
- **Reference test set.**  1 000 Sobol' points per case, shared
  across methods.
- **Hyperparameters.**  Canonical table at
  `benchmark/hyperparameters.py`, pinned by
  `benchmark/tests/test_hyperparameters.py`.  "Fit-per-case" versus
  "pre-registered" categorisation follows Methods §M3.

---

## 4 · Primary and secondary endpoints

| Discovery | Primary endpoint | Secondary endpoints |
|---|---|---|
| D1 | Spearman $\rho$ at $10\,d$ with CI | Kendall $\tau$; rank-shift CI per method; Spearman $\rho$ at $20\,d$ |
| D2 | number of passing topology classes at $m = 10$, NRMSE $\le 0.20$ | sample-complexity curve $n^*(d, \text{topology})$; ditto at $\text{NRMSE} \le 0.05$ |
| D3 | mean signed bias at $10\,d$ budget | per-case breakdown; $F_1$ at the failure boundary; Brier score |

---

## 5 · Planned deviations / exclusions

- Runs with `status != "ok"` (numerical divergence, fit failure) are
  excluded from statistical tests but retained in the raw CSV.
- The formal headline archive must use the complete 20-method
  `nature_pool()` environment.  Diagnostic local environments may expose
  fewer methods if optional dependencies are absent, but
  `scripts/check_server_launch_ready.py` and the formal full pipeline
  require BoTorch/Torch/GPyTorch, formal case backends, and runtime
  monitoring before the server sweep can launch.
- Cluster-aware bootstrap (case-level) is pre-registered as the
  *primary* SE / CI estimator for D3.  If cluster-aware and naïve
  bootstrap disagree on the sign or significance of D3 we will report
  both, with cluster-aware as the headline.

---

## 6 · What would falsify each Discovery

- **D1 falsified** if the upper 95 % CI of $\rho$ at $10\,d$ is
  $\ge 0.70$ **and** both per-method shifts (random+gp, lhs_opt+rbf)
  have CIs including zero.
- **D2 falsified** if $\ge 4$ topology classes pass at $m = 10$,
  NRMSE $\le 0.20$ (the observed pass count is at least 4 of 7; the
  confirmatory test is reported separately through its one-sided upper bound).
- **D3 falsified** if the Wilcoxon two-sided $p$ is $> 0.05 / 3$
  **or** the bootstrap CI includes zero.

The exact decision line for each Discovery, with the 30-repeat
numbers, will be written into `paper/preregistration_decisions.md`
after the sweep, before any manuscript text is amended.

---

## 7 · Pre-registered analysis scripts

The following pipeline stages are executed, in order, on the designated
Linux server.  Each stage is idempotent and deterministic given the
recorded case/repeat seed contract.  Local workstations may run only the
pre-server checks and smoke tests described in
`docs/SERVER_RUN_HANDOFF.md`; they must not launch the full sweep.

```bash
# 0. No-sweep server launch preflight
python scripts/check_server_launch_ready.py

# 1. Formal Nature-grade pipeline on the Linux server
bash scripts/run_pipeline.sh

# Low-level Stage-2 command used by the pipeline and only for reviewed
# server resume/diagnostic work:
python benchmark/scripts/run_full_benchmark.py \
    --cases ALL --pool nature --budgets 5 10 20 50 --n-repeats 30 \
    --seed-strategy case_repeat --seed-start 1000 \
    --confirm-large-run --require-botorch \
    --require-formal-case-backends --require-runtime-monitoring \
    --output-dir benchmark/results/full_benchmark

# 2. Discovery-1 analysis, integrated as pipeline Stage 4
python benchmark/scripts/run_ranking_reversal.py \
    --pool nature --pairing-mode all --n-repeats 30 \
    --output-dir benchmark/results/ranking_reversal
python -m benchmark.scripts.recompute_spearman_ci \
    --rr-dir benchmark/results/ranking_reversal --n-boot 10000

# 3. Discovery-2 analysis, integrated as pipeline Stage 5
python benchmark/scripts/run_sample_complexity.py \
    benchmark/results/full_benchmark/results.csv \
    --threshold 0.20 --rule-budget 10 \
    --output-dir benchmark/results/sample_complexity_0.20

# 4. Discovery-3 analysis, integrated as pipeline Stage 6
python benchmark/scripts/run_reliability_recovery.py \
    benchmark/results/full_benchmark/results.csv \
    --budget-mult 10 --cluster-by case_id \
    --output-dir benchmark/results/reliability_recovery

# 5. Refresh manuscript macros after row-level contracts pass
python scripts/fill_paper_numbers.py \
    --results-csv benchmark/results/full_benchmark/results.csv \
    --rr-dir benchmark/results/ranking_reversal \
    --d2-dir benchmark/results/sample_complexity_0.20 \
    --d3-dir benchmark/results/reliability_recovery \
    --figdata-dir paper/figdata \
    --test-protocol-dir benchmark/results/full_benchmark/analysis \
    --output-tex paper/numbers.tex \
    --output-json paper/numbers.json \
    --require-results-csv \
    --require-results-contract \
    --require-submission-complete
```

---

## 8 · Authorship and contributions at freeze time

- **Corresponding author**: Binglin Wang (conceptualisation, software,
  analysis, writing).
- **To be added**: collaborators with complementary domain expertise
  (structural reliability, aerodynamics, statistics), each logged
  here upon joining.  No co-author may be added after the freeze
  commit without an explicit amendment entry in
  `paper/preregistration_amendments.md`.

---

## 9 · Version history

| Version | Date | Change |
|---|---|---|
| 0.4.0-prereg | 2026-05-01 | Initial freeze prior to the 30-repeat sweep. |
