# Pre-registration amendments

> This document records every deviation from
> `paper/preregistration.md` (frozen at `v0.4.0-prereg` on 2026-05-01).
> The headline 30-repeat sweep — Discoveries 1, 2, 3 — uses the
> **frozen** `nature_pool()` (20 methods) on **all 34 cases** as
> originally pre-registered.  All amendments below are *additional*
> experiments reported in Extended Data; they cannot reach back into
> the headline numbers.

Each entry must include: amendment ID, date (UTC+08:00), scope tag,
rationale, and the file(s) that implement the amendment.  The
amendment IDs are referenced from the manuscript's §Methods so a
reviewer can audit *exactly* what was added after the freeze.

---

## A0 — Operational enforcement of the 20-method headline pool

* **Date**: 2026-06-02
* **Scope**: Launch gate / environment enforcement (does not modify headline)
* **Type**: Pre-flight clarification

### What changed
The frozen protocol allowed an 18-method non-BoTorch fallback if the optional
BoTorch dependency was genuinely unavailable.  For the Nature-family headline
archive we instead require the complete 20-method `nature_pool()` environment
and the optional simulator packages that define formal case response surfaces:
`scripts/run_pipeline.py --profile full` now invokes
`scripts/check_optional_dependencies.py --pool nature --require-botorch
--expected-methods 20 --require-formal-case-backends --require-runtime-monitoring` before the main sweep.

### Rationale
The manuscript, server handoff and acceptance checks all make a 20-method
headline claim.  Treating a missing BoTorch stack as an early launch blocker
prevents the server archive from silently changing the estimand after
submission-facing text has been drafted.  Treating missing NeuralFoil or
AeroSandbox as an early launch blocker likewise prevents `airfoil_stall_2d`
from changing response surface across hosts.

### Files
* `scripts/check_optional_dependencies.py` — optional dependency probe and
  `--require-botorch` / `--require-formal-case-backends` /
  `--require-runtime-monitoring` modes
* `scripts/run_pipeline.py` — formal full-profile preflight wiring
* `docs/SERVER_RUN_HANDOFF.md` — server-operator handoff note

---

## A1 — Post-registration SOTA robustness experiment

* **Date**: 2026-05-09
* **Scope**: Additional methods (does not modify headline)
* **Type**: Robustness check / supplementary baseline

### What changed
We add two state-of-the-art baseline methods to a *separate*
post-registration sweep:

| Method        | Source                                            | Memory class |
|---------------|---------------------------------------------------|--------------|
| `botorch_turbo` | TuRBO (Eriksson et al. 2019, NeurIPS)           | medium       |
| `doptimal+gp`   | D-optimal Fedorov-Mitchell exchange (Atkinson--Donev--Tobias 2007) | light |

These methods live in `benchmark.methods.sota_pool()`; the frozen
`nature_pool()` remains untouched.  Pre-registration compliance test
`benchmark/tests/test_prereg_compliance.py::TestPreregMethods::test_nature_pool_has_20_methods_when_botorch_available`
continues to require `nature_pool() == 20`.

### Rationale
The pre-registration freeze date (2026-05-01) preceded an external
audit which flagged two recurring reviewer concerns:

1. *"Where is TuRBO?"* — TuRBO has been the dominant high-dim BO
   baseline in the BoTorch / NeurIPS community since 2019.  Its
   absence from the headline pool would invite an immediate reviewer
   request.
2. *"Where is the classical D-optimal baseline?"* — D-optimal
   designs are the textbook engineering DOE reference; omitting them
   would be perceived as an oversight by the engineering DOE
   community.

We therefore commit to running an **additional** sweep
(`pool="sota"`, 22 methods on all 34 cases × 4 budgets × 30 repeats)
whose results are reported in Extended Data Table E.1 and discussed
in §Methods (subsection "Additional baselines").  The headline
Discovery numbers remain those of the frozen `nature_pool()` sweep.

### Files
* `benchmark/bo_methods.py` — `BoTorchTurboMethod` class
* `benchmark/samplers.py` — `DOptimalSampler` (Fedorov-Mitchell exchange)
* `benchmark/methods.py` — `sota_pool()` constructor
* `scripts/launch_sota_robustness.sh` — launcher

### Falsification clauses
A1 *does not* alter the falsification clauses for D1, D2, D3 in
§6 of the pre-registration.  The SOTA sweep produces a separate
finding which we will pre-state as:

> **A1-finding**: At budget = 10d, the median rank of TuRBO across
> all 34 cases is among the top-3 methods of the 22-method
> `sota_pool()` *iff* the case has `dim_input >= 8`; otherwise its
> rank is statistically indistinguishable (paired Wilcoxon, Holm
> α = 0.05) from the best static-design method.

We will report A1-finding's outcome in Extended Data Table E.1
*regardless of direction*.

---

## A2 — Multi-fidelity supplementary experiment

* **Date**: 2026-05-09
* **Scope**: Additional methods on a subset of cases
* **Type**: Supplementary experiment

### What changed
The five cases with `supports_multifidelity = True` were not
exercised at multi-fidelity in the pre-registered sweep:

| Case                       | Fidelity levels                | Cost ratio (high/default) |
|----------------------------|--------------------------------|---------------------------|
| `casting_solidification_5d`| `default`, `high`              | ~7×                       |
| `hetero_conduction_2d`     | `default`, `high`              | ~16×                      |
| `laminar_flame_speed_5d`   | `low`, `default`, `high`       | ~8×                       |
| `oil_reservoir_9d`         | `default`, `high`              | ~8×                       |
| `soil_consolidation_5d`    | `default`, `high`              | ~7.5×                     |

We add an MF-Cokriging method (Forrester--Sobester--Keane 2007)
exposed as `benchmark.methods.mf_pool()`.  The MF run uses the
same 4 budgets / 30 repeats protocol but only on the 5 MF-capable
cases.

### Rationale
Demonstrates that the case library's MF capability is non-vacuous
and provides one published-baseline data point for future MF-BO
comparisons against this benchmark.

### Files
* `benchmark/bo_methods.py` — `MFCokrigingMethod`
* `benchmark/methods.py` — `mf_pool()`
* `benchmark/scripts/run_full_benchmark.py` — `--cases-filter mf`
* `scripts/launch_mfbo.sh` — launcher

### Falsification clauses
A2 *does not* alter the falsification clauses for D1, D2, D3.
The MF experiment is reported in Extended Data Figure E.2
(NRMSE-vs-cost-budget Pareto fronts for each of the 5 MF cases).

---

## A3 — D1 pairing-mode robustness check

* **Date**: 2026-05-09
* **Scope**: Methodology robustness (does not modify headline)
* **Type**: Pre-registered sensitivity analysis (now formalised)

### What changed
The `run_ranking_reversal.py` script previously used a single
*dimension*-based pairing of analytical functions to physics cases
(e.g., Branin 2D ↔ TMD 3D).  We now expose three pairing modes via
`--pairing-mode {dimension, topology, random}` and recommend running
all three for the §Robustness paragraph of Discovery 1:

* `dimension` (default, identical to the pre-registered behaviour)
* `topology`  — match by Jaccard similarity of `response_topology` tags
* `random`    — random permutation, used as a null distribution

### Rationale
A reviewer comment at the freeze read: *"Branin (smooth) is paired
with TMD (threshold) — the pairing itself drives the rank divergence
finding."*  Running all three modes lets us answer this concern
explicitly: if Spearman ρ stays below the 0.7 threshold under the
**dimension** and **topology** pairings but rises under **random**,
the rank-divergence finding is robust to pairing choice.

### Files
* `benchmark/scripts/run_ranking_reversal.py` — three pairing helpers
  + `_run_one_pairing` driver
* Outputs go to `benchmark/results/ranking_reversal/pairing_<mode>/`

### Falsification clauses
The headline D1 falsification clause (§6 of the pre-registration)
remains unchanged.  We *additionally* commit:

> **A3-finding**: D1 is reported as *robust to pairing* iff
> Spearman ρ at 10d *both* under `dimension` *and* under `topology`
> falls below the upper bound of 0.70, while ρ under `random`
> exceeds the corresponding ρ at the smallest budget.

If A3-finding is violated we will report it explicitly and shift
D1 from "confirmed" to "confirmed but pairing-sensitive".

---

## A4 — D2 confidence-bound terminology clarification

* **Date**: 2026-08-31
* **Scope**: Definitional clarification (does not modify headline)
* **Type**: Statistical wording correction

### What changed
The D2 protocol and statistical-analysis plan now consistently call the
registered quantity the **one-sided Clopper--Pearson upper confidence bound**
on the topology pass rate.  The rejection rule is unchanged: reject the null
only when this upper bound is below $4/7$.  The separate operational
falsification statement remains the observed condition of at least 4 of 7
topology classes passing.

### Rationale
Earlier prose used "lower bound" while simultaneously describing an upper
bound decision rule.  The implementation and materialized result already used
the upper bound (`clopper_pearson_one_sided_upper95 = 0.7746784`); this entry
records the terminology correction without changing data, code, thresholds,
or decisions.

### Files
* `paper/preregistration.md`
* `docs/STATISTICAL_ANALYSIS_PLAN.md`
* `scripts/generate_si.py`
* `scripts/check_paper_draft_ready.py`
* `paper/si_multiple_testing.{md,tex}`

### Falsification clauses
No clause, threshold, estimator, or result was changed.

---

## A5 — D3 estimand and test alignment clarification

* **Date**: 2026-08-31
* **Scope**: Definitional clarification (does not modify headline)
* **Type**: Statistical wording correction

### What changed
The D3 documentation now distinguishes the pooled mean signed bias, which is
the reported effect-size estimate and receives the cluster-bootstrap interval,
from the Wilcoxon signed-rank test, which tests positive distributional
location (operationally, median signed bias).  The decision rule, data, test,
bootstrap procedure, thresholds and Holm correction are unchanged.

### Rationale
Earlier prose used expectation/mean language for a Wilcoxon test.  That wording
could be read as a direct test of the mean, which the signed-rank statistic does
not provide.  The clarification makes the two estimands explicit: at the 10d
budget the pooled mean is 0.0263702 and the per-cell median is 0.0040.  The
registered estimand and decision rule are unchanged.

### Files
* `paper/preregistration.md`
* `docs/STATISTICAL_ANALYSIS_PLAN.md`
* `paper/preregistration_decisions.{md,template.md}`
* `paper/nature_cs_main.tex`
* `paper/reporting_summary.md`

### Falsification clauses
No clause, threshold, estimator, or result was changed.

---

## A6 — D3 probability-grid precision normalization

* **Date**: 2026-08-31
* **Scope**: Additional methods (does not modify headline interpretation)
* **Type**: Numerical reproducibility correction

### What changed
The D3 signed-bias derivation now rounds probability differences to 12 decimal
places before aggregation and explicitly excludes non-`ok` rows. This removes
parser-dependent binary floating-point tie behavior for the archived
one-thousandth probability grid. The current pooled values are mean
`0.0263702194`, registered case-cluster percentile CI
`[0.0017058516, 0.0592297917]`, and one-sided Wilcoxon `p=1.4327519e-251`.

### Rationale
The previous calculation could produce different Wilcoxon statistics when the
same CSV was parsed by pandas versus Python's CSV reader, despite identical
decimal values. Canonical decimal normalization makes the derivation
deterministic without changing the archived rows, budget, status policy,
bootstrap seed, or scientific direction.

### Files
* `benchmark/reliability_recovery.py`
* `scripts/derive_pooled_reliability_stats.py`
* `scripts/derive_d3_case_level_sensitivity.py`
* `paper/numbers.{json,tex}`
* `paper/preregistration_decisions.md`

### Falsification clauses
No clause, threshold, or headline interpretation was changed.

---

## A7 — D3 positive-direction Holm correction

* **Date**: 2026-09-01
* **Scope**: Definitional/statistical implementation clarification (does not modify headline data)
* **Type**: Bug fix with transparent sensitivity retention

### What changed

The D3 implementation now applies the registered Holm correction to the
positive one-sided Wilcoxon $p$-values (`signed_bias_pvalue_pos`), matching the
registered alternative $\operatorname{median}(b)>0$. The two-sided and negative
one-sided $p$-values remain in the artifact as sensitivity fields. Downstream
figure, SI and number-generation scripts now consume the same positive-direction
column. The corrected maximum method-level adjusted $p$ is `1.56e-4` (the
previous `3.10e-4` was the adjustment of the two-sided sensitivity values).

### Rationale

The earlier artifact used the two-sided value for the `p_holm` column despite
describing the primary D3 test as one-sided. The raw signed-bias rows, effect
sizes, confidence intervals, test direction, and the set of 20 significant
methods are unchanged. This entry records the correction so the frozen
pre-registration, implementation and manuscript remain auditable.

### Files

* `benchmark/reliability_recovery.py`
* `scripts/prepare_figdata.py`
* `scripts/fill_paper_numbers.py`
* `scripts/check_submission_ready.py`
* `paper/preregistration_decisions.md`
* `docs/STATISTICAL_ANALYSIS_PLAN.md`
* `paper/nature_cs_main.tex`
* `paper/si_d3_per_method.tex`

### Falsification clauses

No threshold, data row, estimator, or headline decision was changed. The
secondary two-sided test remains available for sensitivity reporting; the
primary decision continues to use the positive one-sided test plus the
case-cluster interval for the mean effect.

---

## A8 — SAASBO NUTS hyperparameter documentation correction

* **Date**: 2026-09-14
* **Scope**: Documentation correction only (does not modify headline data or decisions)
* **Type**: Bug fix (documentation vs. executed configuration)

### What changed
`benchmark/hyperparameters.py` documented `warmup_steps=128`, `num_samples=64` and
`thinning=16` for `botorch_saasbo`, and Supplementary Table S1 reproduced those
numbers.  The archived sweep ran 4/4/1.  `RUN_MANIFEST.json` records the resolved
constructor configuration of every component the sweep instantiated, and its
`botorch_saasbo` entry carries `warmup_steps=4`, `num_samples=4`, `thinning=1`.
The registry and the table now state 4/4/1.  A provenance sentence that justified
the old values by a "30--60 s per NUTS step" cost was removed along with them.

### Rationale
A hyperparameter table has to describe the run that produced the archive.  The
sweep cannot be re-run and `results.csv` is cryptographically pinned, so
re-aligning the *documentation* is the only integrity-preserving direction.  The
consistency check previously walked only `full_method_pool()`, i.e. the 18
always-present methods, which is why the optional BoTorch entry could drift
unnoticed; it now also walks `bo_method_pool()`.
`benchmark/tests/test_hyperparameters.py` gained three guards: pool coverage and
live-kwarg equality, byte-pinning of the method sources against the manifest, and
a key-by-key comparison of the documented registry against the frozen manifest
configuration (49 documented keys; verified to fail if the 128/64/16 values are
reintroduced).

### Files
* `benchmark/hyperparameters.py`
* `benchmark/tests/test_hyperparameters.py`
* `paper/si_hyperparameters.{md,tex}`
* `paper/si_hyperparameters_clean.tex`
* `docs/REVISION_20260914_FIXES.md`

### Falsification clauses
No clause, threshold, estimator, data row or headline decision was changed.  The
archived accuracy of `botorch_saasbo` is unaffected by restating its settings.

---

## A9 — D1 Kendall tau reported as registered; decision worksheet completed

* **Date**: 2026-09-14
* **Scope**: Reporting completeness (does not modify the D1 decision)
* **Type**: Definitional clarification / reporting correction

### What changed
The pre-registration registers Kendall tau as the D1 sensitivity statistic
alongside Spearman rho, with the matching bootstrap interval, and requires the two
measures to agree in sign and order.  `correlation.json` already contained both
(the primary dimension-matched $10d$ record gives $\tau = 0.7729730$ with a
percentile interval of $[0.6041255, 0.9019926]$), but the value reached neither
`numbers.json` nor the manuscript.  It is now exported as
`dOneKendallTau{,CILo,CIHi}`, reported in the main-text D1 paragraph, tabulated
for all six pairing--budget combinations in Supplementary Table S13, and entered in
the D1 worksheet of `paper/preregistration_decisions.md` under the template's
"Secondary (Kendall tau)" heading.

### Rationale
An unregistered omission of a registered analysis is a reporting defect even when
the underlying value is archived: the reader cannot distinguish "measured and
consistent" from "never checked".  The manuscript now also states explicitly that
the tau interval is wider than the rho interval and that its lower bound falls
below the $0.70$ threshold, so the sensitivity statistic corroborates the
direction of the D1 headline rather than restating its full strength.  The D1
decision, thresholds and estimators are unchanged.

### Files
* `scripts/fill_paper_numbers.py`
* `scripts/review_response_analyses.py`
* `paper/nature_cs_main.tex`
* `paper/supplementary.tex`
* `paper/si_d1_tau.tex`
* `paper/preregistration_decisions.md`
* `paper/numbers.{json,tex}`

### Falsification clauses
No clause, threshold, estimator or result was changed.

---

## Format for future amendments

```
## A<N> — <short title>
* **Date**: YYYY-MM-DD
* **Scope**: <Headline | Additional methods | Robustness | Other>
* **Type**: <Robustness check | Bug fix | Definitional clarification | Additional experiment>

### What changed
…

### Rationale
…

### Files
…

### Falsification clauses
…
```

Append-only.  Do not edit historical entries.
