# Glossary

> **Audience.**  Readers from outside the DOE / surrogate-modelling
> community who hit an unfamiliar term while reading the docs, the
> code, or the paper.  Terms are grouped, then alphabetised within each
> group.  Where a term has a direct entry-point in the codebase, the
> file path is given.

## 1. Project-specific names

### DOE-Benchmark
The whole study: 34 physics-grounded engineering test cases × 20 DOE /
surrogate / Bayesian-optimisation methods × 4 budgets × 30 repeats,
with three pre-registered Discoveries (D1 / D2 / D3).

### Discovery 1 (D1)
"Are method rankings on **analytical** functions predictive of method
rankings on **physics-grounded** engineering cases?"  See
[`paper/preregistration.md`](../../manuscripts/protocol/preregistration.md) §1.1 for the
H₀ / H₁ statement.  Implementation: `benchmark/scripts/run_ranking_reversal.py`.

### Discovery 2 (D2)
"Does the textbook *n = 10·d* rule (Loeppky, Sacks & Welch 2009) hold
across response-surface topologies on physics-grounded cases?"  See
preregistration §1.2.  Implementation: `benchmark/sample_complexity.py`.

### Discovery 3 (D3)
"On cases that declare a `failure_threshold`, do surrogate-predicted
failure probabilities $\hat P_f^{\text{surr}}$ deviate systematically
from the truth $\hat P_f^{\text{true}}$?"  See preregistration §1.3.
Implementation: `benchmark/reliability_recovery.py`.

### Stage 0 → 10
The 11-step pipeline that takes the project from a smoke test to a
fully-archived submission.  Stages 0–7 are sweep + analysis; Stages 8–9
are figures; Stage 10 is the reproducibility archive.  Full walkthrough:
[`docs/REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

### `nature_pool()` / `sota_pool()` / `mf_pool()`
Method-pool factories in `benchmark/methods.py`.  `nature_pool()` is
the **frozen** pre-registered 20-method pool used for D1 / D2 / D3.
`sota_pool()` adds TURBO + D-optimal (amendment A1).  `mf_pool()` adds
multi-fidelity Cokriging (amendment A2).

### Pre-registration
The frozen scientific protocol document [`paper/preregistration.md`](../../manuscripts/protocol/preregistration.md)
that fixes the hypotheses, decision rules, and falsification clauses
**before** the headline sweep produces results.  Labelled
`v0.4.0-prereg`; that label is a version identifier, and the
freeze date is an author attestation rather than a resolvable
git tag (see the freeze-provenance note in that document).

### Pre-registration amendments
Append-only log [`paper/preregistration_amendments.md`](../../manuscripts/protocol/preregistration_amendments.md)
of post-freeze deviations.  Three amendments to date: A1 (SOTA pool),
A2 (multi-fidelity), A3 (D1 pairing-mode robustness).

### `RUN_MANIFEST.json`
Stage-10 immutable archive emitted by
`scripts/generate_run_manifest.py`.  Captures the git commit, package
versions, machine identity, and SHA-256 of every artefact in
`benchmark/results/full_benchmark/`.

## 2. DOE / surrogate-modelling vocabulary

### Active learning
A sequential design strategy where the next training point is chosen
based on the current surrogate's uncertainty / acquisition value.
Examples in this benchmark: EGO/EI, Max-Variance, AK-MCS.

### AK-MCS (Active learning Kriging — Monte Carlo Simulation)
A reliability-focused active learning algorithm by Echard, Gayton &
Lemaire (2011).  Adaptively places samples near the failure boundary.

### Brier score
Mean squared error between a probabilistic prediction and a binary
outcome.  Used here as one of the per-cell reliability metrics
(`f_brier_o{j}` in `results.csv`).

### Budget multiplier (`m` or `budget_mult`)
Training set size expressed as a multiple of input dimension `d`:
`m = n / d`.  Pre-registered values are `m ∈ {5, 10, 20, 50}`.  The
"10·d rule" in D2 corresponds to `m = 10`.

### Cliff's δ
A non-parametric effect-size measure for paired ordinal data.  Reported
alongside Wilcoxon p-values in `benchmark/rigour.py` (with bootstrap
CI).

### Critical Difference (CD)
The post-hoc threshold for the Friedman–Nemenyi rank test:
`CD = q_α · √(k(k+1) / (6N))` for `k` methods on `N` cases.  Two
methods are **not** statistically different if their mean-rank gap is
below CD.  Implementation: `benchmark.analysis.critical_difference()`.

### `d` (input dimension)
Number of input variables of a case.  In this benchmark `d ∈ [2, 20]`.

### D-optimal design
A classical DOE that maximises the determinant of the Fisher
information matrix; minimises the volume of the parameter-estimate
joint confidence ellipsoid.  Available via the `sota_pool()` (A1).

### EGO / Expected Improvement (EI)
*Efficient Global Optimisation* (Jones, Schonlau & Welch 1998).  A
sequential design that picks the next point to maximise expected
improvement under a Gaussian-process surrogate.

### Friedman test
A non-parametric omnibus test for differences across `k` methods on
`N` matched samples.  Pre-condition for Nemenyi post-hoc.

### GP / Gaussian Process
A non-parametric surrogate that models the function as a multivariate
Gaussian; provides both mean and variance predictions.  Several
variants in this suite: GP-Matern, GP-RBF, SAASBO (sparse-axis-aligned
Bayesian).

### Halton sequence
A low-discrepancy sequence using prime-base reflection; an alternative
to Sobol' for quasi-Monte-Carlo sampling.

### Holm-Bonferroni correction
Step-down family-wise error control for multiple testing: order
p-values ascending, reject H₍ⱼ₎ iff `p₍ⱼ₎ ≤ α / (k − j + 1)`.  Used
across the three Discoveries (k=3) and within D3 across methods.
Implementation: `benchmark.reliability_recovery._holm_adjust()`.

### Kendall τ
A rank-correlation coefficient based on concordant vs discordant
pairs.  Reported alongside Spearman ρ in D1 as the secondary
correlation measure.

### LHS / Latin Hypercube Sampling
A space-filling design that ensures one-dimensional projections are
uniform.  This benchmark uses two variants: vanilla LHS and
maximin-optimised LHS (`OptimisedLHSSampler`).

### Multi-fidelity (MF)
Fidelity hierarchy where a cheap-low-fidelity model and an
expensive-high-fidelity model are combined (e.g. via Cokriging).
A2 amendment exercises this on five cases that declare
`supports_multifidelity=True`.

### Nemenyi post-hoc
Pairwise post-hoc test following Friedman: declares two methods
different if their mean-rank gap exceeds CD (see *Critical Difference*).

### NRMSE / Normalized Root-Mean-Square Error
For each output, `NRMSE = RMSE / sample_std(y_true)` on the held-out test set
(`ddof=1` in `benchmark/metrics.py`). The benchmark averages the three
per-output values for its headline accuracy score. This is not range
normalization; a constant held-out output has undefined NRMSE.

### NLL / Negative Log-Likelihood
`-log p(y_true | model)` evaluated on the test set.  Reported for
methods that produce calibrated predictive distributions (GPs).

### `n*` / `n_star`
The smallest `n` (training set size) at which a method's median NRMSE
falls under a target threshold (default 0.20).  Often expressed as
`n*/d` so it can be compared across cases of different dimension.

### `pf_pred` / `pf_true`
Predicted vs true failure probability on a 1 000-point Sobol test set,
output-by-output (`pf_pred_o0`, `pf_pred_o1`, ...).  Computed by
`benchmark/reliability.py`.

### PICP / Mean Prediction Interval Coverage Probability
Fraction of test points that fall inside the surrogate's predicted
interval at a given confidence level.  Used as a calibration metric.

### Polynomial Chaos
A surrogate that expands the response in orthogonal polynomial bases
(Legendre, Hermite, ...) of the inputs.  Implemented via `chaospy`.

### Random Forest (RF)
Ensemble of regression trees.  Non-Gaussian; cannot extrapolate
outside the training-data envelope (relevant to D3 reliability bias).

### RBF / Radial Basis Function surrogate
Interpolant of the form `y(x) = Σ wᵢ φ(‖x − xᵢ‖)`.  Pure
interpolator (passes through every training point) — different from
GP-RBF which uses an RBF kernel.

### SAASBO (Sparse Axis-Aligned Subspace BO)
A high-dimensional BoTorch BO method that places horseshoe-prior
sparsity on the GP length-scales.  Most memory-intensive method in
this suite.

### Sobol' sequence
A quasi-random low-discrepancy sequence.  Used for the test-set
construction (1 000 points per case, shared across methods) and as
one of the candidate samplers.

### Sobol' indices ($S_T$, $S_1$)
Variance-based sensitivity indices.  $S_1$ is the first-order
contribution; $S_T$ is the total contribution.  Recovered via Saltelli
on the fitted surrogate in Stage 7 (sensitivity recovery).

### Spearman ρ
Rank-correlation coefficient: Pearson correlation on rank-transformed
inputs.  Primary D1 statistic.  Bootstrap CI computed by
`benchmark/scripts/recompute_spearman_ci.py`.

### TURBO (Trust-Region BO)
A high-dimensional BO method (Eriksson et al. 2019) that maintains a
shrinking trust region.  Available in `sota_pool()` via A1.

### Wilcoxon signed-rank test
A non-parametric paired test on the differences `d_i = b_i`.  Used in
D3 to test the signed-bias direction (`alternative ∈
{less, greater, two-sided}`).

## 3. Statistics-of-the-experiment vocabulary

### Bootstrap CI (percentile method)
Resample the data with replacement `n_boot` times, compute the
statistic on each resample, take the 2.5 / 97.5 percentiles.  Used
throughout for CIs on means, ρ, τ, and rank shifts.

### Cluster bootstrap
Bootstrap variant where the unit of resampling is a *cluster*, not an
individual observation.  Important for D3 where 30 repeats per case
are not independent.  Activated via `--cluster-by case_id`.

### Effect size
A measure of practical significance separate from p-value: how big is
the difference?  Examples: Spearman ρ (D1), proportion-of-passes (D2),
mean signed bias (D3), Cliff's δ (rigour).

### Family-wise error rate (FWER)
Probability of at least one Type-I error across a family of related
tests.  Controlled here at α=0.05 across {D1, D2, D3} via Holm.

### Falsification clause
Pre-registered set of conditions that, if observed, force the
manuscript to report the Discovery as falsified rather than confirmed.
See [`paper/preregistration.md`](../../manuscripts/protocol/preregistration.md) §6.

### Pre-specified vs post-hoc
Pre-specified analyses are listed in the SAP before any data is seen.
Post-hoc analyses are added after looking at results — they cannot
be reported as confirmatory and require a `preregistration_amendments`
entry.  See [`docs/STATISTICAL_ANALYSIS_PLAN.md`](STATISTICAL_ANALYSIS_PLAN.md).

### `seed_strategy = case_repeat`
Per-cell seed derivation that ensures the *same training set* is
shown to every method on the same `(case_id, repeat)` cell, preserving
the paired-sample structure required by Friedman / Wilcoxon.  The
alternative `legacy` strategy uses `seed_start + repeat` (no case
binding).  Implementation: `benchmark.core.derive_seed`.

## 4. Engineering-domain vocabulary

### Failure threshold (`failure_threshold`)
A scalar value declared on a `CaseMeta.output_variables[j]` entry that
defines a binary failure event: an output sample fails iff `y > thr`
(or `< thr`, depending on `failure_direction`).  16 of the 34 cases
declare a failure threshold; D3 is computed only on those.

### Failure direction (`above` / `below`)
Whether the failure event is `y > thr` (`above`) or `y < thr`
(`below`).  Required to interpret `failure_threshold` correctly.

### Response-feature label (`response_topology`)
Qualitative classification of the response surface: `smooth_unimodal`,
`smooth_multimodal`, `smooth_monotone`, `strong_interaction`,
`sparse_effects`, `discontinuous`, `threshold_jump`, `heterogeneous`.
A case may carry multiple response-feature labels, so the resulting case
sets are called response-feature groups in the manuscript.  Used to
stratify D2.  The case-registry field name remains `response_topology`.

### Test-set
The shared 1 000-point Sobol' reference set per case.  Frozen so all
methods are evaluated on the same points (paired comparison).

### VS scenario (VS-A through VS-H)
"Validation Scenario" tags on the case library that classify each case
by which DOE-method axis it stresses (e.g. high-dim, multi-modal,
threshold).  Used by `benchmark/stratified.py` for diagnostic
breakdowns.

## 5. Code / infrastructure vocabulary

### `--allow-deviation`
CLI flag on `validate_run_config.py` (and forwarded by `run_pipeline`)
that permits a non-pre-registered configuration to launch.  Logs to
`preregistration_deviations.txt`.

### `--skip-existing`
CLI flag on `run_full_benchmark.py` and `run_ranking_reversal.py`:
resume from a partial `results.csv`.  Re-running the same command is
idempotent and continues from the last successful cell.

### CVD-safe (color-vision-deficiency safe)
Plotting requirement enforced by Nature Methods 2011 (Wong, "Points of
view: Color blindness").  This project uses Wong's 8-colour palette
+ `cividis` for heatmaps; tests in `benchmark/tests/test_paper_pipeline.py`
guard against re-introduction of unsafe colormaps.

### figdata
The CSV layer between `results.csv` and the final figures.  Generated
by `scripts/prepare_figdata.py`; consumed by `scripts/plots_nature.py`
and `scripts/fill_paper_numbers.py`.  See
[`docs/REPRODUCIBILITY.md`](REPRODUCIBILITY.md) §10.

### figure manifest
`paper/figure_manifest.json` — emitted by `scripts/plots_nature.py`,
records SHA-256 of every PDF + PNG + figdata input so reviewers can
trace any figure pixel back to a raw data file.

### One-click pipeline
`scripts/run_pipeline.{py,ps1,sh}` — the single command that runs
Stage 0 → 10 in order, verifies expected outputs between stages, and
streams logs.  See [`scripts/README.md`](../scripts/README.md).

## 6. References for further reading

| Topic | Reference |
|---|---|
| Friedman + Nemenyi + CD | Demšar (2006), *JMLR* 7:1–30 |
| Pre-registration | Nosek et al. (2018), *PNAS* 115:2600 |
| 10·d rule | Loeppky, Sacks & Welch (2009), *Technometrics* 51:366 |
| Holm correction | Holm (1979), *Scand. J. Stat.* 6:65 |
| Wilcoxon signed-rank | Wilcoxon (1945), *Biometrics* 1:80 |
| Bootstrap | Efron & Tibshirani (1993), *An Introduction to the Bootstrap* |
| EGO / EI | Jones, Schonlau & Welch (1998), *J. Glob. Optim.* 13:455 |
| AK-MCS | Echard, Gayton & Lemaire (2011), *Struct. Saf.* 33:145 |
| TURBO | Eriksson et al. (2019), *NeurIPS* |
| SAASBO | Eriksson & Jankowiak (2021), *UAI* |
| Sobol' indices (Saltelli) | Saltelli et al. (2010), *Comput. Phys. Commun.* 181:259 |
| CVD-safe colours | Wong (2011), *Nat. Methods* 8:441 |
