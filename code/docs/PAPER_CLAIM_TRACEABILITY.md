# PAPER CLAIM TRACEABILITY — DOE-Benchmark

This is a dated evidence map from the 2026-09-14 audit, not a current authoring
or submission checklist. The quantitative checks below remain tied to the frozen
data. In the public repository, code lives in `code/`; retained figure data,
supplementary tables and protocols live in `manuscripts/`. Historical `paper/`
paths below refer to the development manuscript tree.

The 2026-09-26 post-hoc practitioner guidance and Supplementary Table S14 have
a separate reproducible audit in
`project_docs/manuscript_revision_20260926/design_model_guidance/` from
the repository root. Run `scripts/explore_design_model_guidance.py` from
`code/` to regenerate its case-level medians, group ranks, paired contrasts
and `manuscripts/tables/si_design_model_guidance.tex`. The 21 metadata-labelled
strong-interaction cases are not the same group as the 23 Sobol'-threshold
interaction-dominated cases used elsewhere; the five sparse-effects cases
overlap with other topology labels. These patterns are descriptive, not
prespecified or factorial design-by-model effects.

Quantitative claims in `paper/nature_cs_main.tex` and the Supplementary
Information, mapped to the code that computes it and the archived data file that
supports it. Purpose: expose each reported result alongside its code path and retained data.

**How to read the "Verdict" column.** *Verified* = the artifact exists, the
script that writes it exists, and (where checked) recomputation reproduced the
reported value. *Regenerated* = re-derived during the 2026-09-14 audit.

**Historical verification commands (run from `02_code/`; `paper/` is a junction to the physical `01_manuscripts/` directory)**

```powershell
python scripts/check_submission_ready.py paper      # manuscript/manifest/numbers/artifacts
python scripts/fill_paper_numbers.py                # rebuild paper/numbers.tex
python scripts/plots_nature.py --figdata paper/figdata --out paper/figures
python scripts/derive_pooled_reliability_stats.py   # D3 pooled + D2 count test
python scripts/build_supplementary.py               # SI tables S1,S2,S3,S6-S9
python scripts/review_response_analyses.py          # SI tables S10-S13
python -m pytest tests benchmark/tests -q
```

## 0. The single upstream artifact

| Item | Value |
|---|---|
| File | `benchmark/results/full_benchmark/results.csv` |
| Rows | 81,600 = 34 cases × 20 methods × 4 budgets (5d/10d/20d/50d) × 30 repeats |
| SHA-256 | `71e72d937b64abc6f5c44da2f6b2183e293e228258ce4c0abce9bb4100f0ca0b` |
| Provenance | `RUN_MANIFEST.json` (environment, case/method fingerprints, artifact hashes) + `run_config.json` |
| Written by | `benchmark/scripts/run_full_benchmark.py` (Stage 2) |

Everything below derives from this file (or from the D1 analytical phase). No
result is hand-typed: `paper/numbers.tex` and `paper/numbers.json` record the
SHA-256 of each input they consumed, and all five declared hashes **match**
their files on disk.

---

## 1. Methods and scope claims

| Claim in manuscript | Code | Data | Verdict |
|---|---|---|---|
| 34 physics-grounded cases, six domains | `doe_testbed/registry/registry.py`, 34 case modules under `doe_testbed/cases/` | `paper/figdata/fig1_case_catalog.csv`; `docs/case_validation_status.csv` | Verified (registry enumerated: 34 cases, domains 8/7/6/5/4/4) |
| Input dimensions 3–20, median 5 | case `CaseMeta.dim_input` | same | Verified (min 3, max 20, median 5.0) |
| Every case returns three outputs | `CaseMeta.dim_output` | same | Verified (all 34 have `dim_output == 3`) |
| 20 methods: 15 static + 5 sequential | `benchmark/methods.py::nature_pool()` | `paper/figdata/fig1_method_catalog.csv` | Verified (`Counter`: static_design 15, five sequential families 1 each) |
| 16 cases expose a failure boundary → D3 | `CaseMeta.has_failure_boundary` | `benchmark/results/reliability_recovery/reliability_long.csv` | Verified (16 True; 16 cases in D3) |
| 23 cases interaction-dominated (ΣS_T > 1.05) | `scripts/cross_validate_sobol.py`, `benchmark/data/sobol_corrected_full_current_source.json` | `paper/tables/si_sobol.tex` column 7 | Verified (23 of 34 rows > 1.05) |
| Four budgets, 30 repeats, 1,000 test points | `benchmark/scripts/run_full_benchmark.py`, `benchmark/core.py::derive_seed` | `run_config.json`, `results.csv` (columns `budget_mult`, `repeat`, `n_test`, `test_seed`) | Verified (n_test=1000, test_seed=0, seed_start=1000 uniform) |
| Evidence tiers: 17/17 P1/P2 | `scripts/generate_case_validation_summary.py` | `docs/case_validation_status.csv`, `paper/tables/si_cases_clean.tex` | Verified (17+3+2+12 tier rows; P1 17 / P2 17) |
| Fig. 1 scope panel | `scripts/plots_nature.py::plot_fig1_scope` | `paper/figdata/fig1_{case,method}_catalog.csv` | **Regenerated 2026-09-14** (family labels fixed) |

---

## 2. D1 — method-rank transfer

| Claim | Code | Data | Verdict |
|---|---|---|---|
| Primary ρ = 0.914, 95% CI [0.758, 0.971] (dimension-matched, 10d) | `benchmark/scripts/run_ranking_reversal.py`; CI by `benchmark/scripts/recompute_spearman_ci.py`; ranks by `scripts/review_response_analyses.py::method_ranks` | `benchmark/results/ranking_reversal/correlation.json`; `paper/figdata/fig2_d1_spearman.csv` | Verified (0.91352; CI [0.7579, 0.9706]) |
| Registered Kendall τ = 0.773, 95% CI [0.604, 0.902] (dimension-matched, 10d); τ positive in all six pairings and every τ interval excludes zero | `benchmark/scripts/recompute_spearman_ci.py` (registered method-axis bootstrap) | `benchmark/results/ranking_reversal/correlation.json`; `paper/tables/si_d1_tau.tex` (S13) | Verified (τ 0.7407–0.8226 across the six pairings; lowest τ CI bound 0.5556) |
| ρ range 0.90–0.94 over six pairings; lowest CI bound 0.73 | same | `correlation.json` (6 records) | Verified (0.8999–0.9430; min CI lo 0.7282) |
| Effective physical-case counts 4/6, 5/6, 5/6 (10d); 4/6, 5/6, 5/6 (20d) | `recompute_spearman_ci.py` | `correlation.json` field `n_physical_cases_complete` | Verified (4,4,5,5,5,5) |
| Rank-policy sensitivity changes ρ by at most **0.044** | `scripts/review_response_analyses.py::method_ranks` (penalized vs complete-case) | `paper/review_response_numbers.json::d1_policy_rhos` | **Corrected on 2026-09-14** (was stated as 0.012; recomputed max 0.0440) |
| Leave-one-function-out ρ between 0.838 and 0.931; global minimum 0.838 | `scripts/review_response_analyses.py::d1_loo` | `benchmark/results/ranking_reversal/d1_loo_sensitivity.csv`; `paper/tables/si_d1_loo.tex` (S10) | Verified (min 0.83804 = dimension/20d minus Friedman; max 0.93137) |
| Three leading methods always Gaussian-process based; at most two names differ; max shift 4.6 ranks; mean shift 1.4–1.7 | `run_ranking_reversal.py` | `pairing_*/rank_comparison.csv` | **Corrected on 2026-09-14** (dimension/10d differs in 2 of 3 names, not 1; max |shift| 4.600, mean 1.383–1.725) |
| D1 panel spans only 3–10-D; 16/20-D cases do not enter D1 | pairing definitions in `benchmark/scripts/run_ranking_reversal.py` | `pairing_*/physical_results.csv` | **Corrected on 2026-09-14** (previous sentence wrongly implied 11–20-D cases were analysed) |
| Fig. 2 rank-transfer panels | `scripts/plots_nature.py::plot_fig2_d1` | `paper/figdata/fig2_d1_{spearman,ranks}.csv` | Verified (labels 0.91/0.94/0.94/0.91/0.92/0.90 match CSV) |

---

## 3. D2 — the 10-samples-per-dimension rule

| Claim | Code | Data | Verdict |
|---|---|---|---|
| 3 of 7 topologies pass at 10d (43%) | `benchmark/scripts/run_sample_complexity.py` | `benchmark/results/sample_complexity_0.20/rule_summary.csv` | Verified (3/7 main, non-exploratory) |
| Clopper–Pearson one-sided upper bound 0.7747 | `scripts/derive_pooled_reliability_stats.py::cp_upper_d2` | `sample_complexity_0.20/tenx_rule_cp_bound.json` | Verified (0.774678) |
| Median n* = 15d across the seven classes | `run_sample_complexity.py` | `rule_summary.csv` `median_n_star` | Verified |
| 55–60% of methods pass on smooth-unimodal / discontinuous / strong-interaction | same | `rule_summary.csv` `frac_pass_at_10d` | Verified (0.55, 0.55, 0.60) |
| 0% pass on smooth-monotone, heterogeneous, sparse-effects; required budget 20d–50d | same | `rule_summary.csv` | Verified (frac 0.00 for all three; median n* 20/50/50) |
| Seven-class exact test inconclusive | `tenx_rule_cp_bound.json` + decision rule in `paper/preregistration.md` | same | Verified |
| Fig. 3 learning curves and per-topology adequacy | `scripts/plots_nature.py::plot_fig3_d2` | `paper/figdata/fig3_d2_{learning,rule}.csv` | Verified |

---

## 4. D3 — failure-probability calibration

| Claim | Code | Data | Verdict |
|---|---|---|---|
| 20/20 tested methods significant after Holm; 0 excluded for low n | `benchmark/scripts/run_reliability_recovery.py` | `benchmark/results/reliability_recovery/reliability_per_method.csv`; `paper/tables/si_d3_per_method.tex` (S6) | Verified (20 rows, all `p_holm` significant) |
| Pooled mean signed bias 0.0264, 95% CI [0.0017, 0.0592] (9,570 runs, 16 cases) | `scripts/derive_pooled_reliability_stats.py::pooled_d3`; extraction in `benchmark/reliability_recovery.py` | `reliability_long.csv` → `pooled_reliability_summary.json` | Verified (0.026370; [0.001706, 0.059230]; n_runs 9570, n_cases 16) |
| BCa interval [0.0057, 0.0692]; leave-one-case-out 0.0138–0.0320 | `run_reliability_recovery.py`, `paper/tables/si_d3_loco.tex` (S11) | `d3_loco_bca.json` | Verified |
| 12/16 case means positive; equal-case CI [0.0015, 0.0585]; Wilcoxon p = 0.0253 > 0.0167 | `scripts/derive_d3_case_level_sensitivity.py` | `d3_case_level_sensitivity.{csv,json}`; `paper/tables/si_d3_case_level.tex` (S8) | Verified (12 positive, p 0.025330, threshold 0.016667) |
| Cluster CI is 3.3–5.7× wider than i.i.d. | `run_reliability_recovery.py` | `reliability_ci_sensitivity.csv`; `paper/tables/si_d3_ci_sensitivity.tex` (S7) | Verified (min ratio 3.269, max 5.647) |
| +0.026 bias over-states P_f by ≈7% of the typical reference value (median P_f ≈ 0.39 at the driving output) | `benchmark/reliability_recovery.py` (`output_aggregate="max"` selects the driving output) | `results.csv` columns `pf_true_o{0,1,2}`, `pf_pred_o{0,1,2}` | **Corrected on 2026-09-14** (median at driving output 0.392 → 6.7%; previous text used 0.52 = output 0 only, and "5–6%") |
| Bias smallest for GP methods (0.012–0.019), largest for GB/linear/kNN (0.056–0.058) | `run_reliability_recovery.py` | `reliability_per_method.csv` | Verified (GP range 0.01209–0.01895; GB 0.05589, linear 0.05598, kNN 0.05842) |
| Test-set Monte-Carlo noise floor | `benchmark/scripts/generate_report.py` | `full_benchmark/analysis/test_set_sufficiency.csv`; `paper/tables/si_mc_noise.tex` (S9) | Verified |
| Fig. 4 per-method bias | `scripts/plots_nature.py::plot_fig4_d3` | `paper/figdata/fig4_d3_bias.csv` | Verified |
| Fig. 5 mechanism view | `scripts/plots_nature.py::plot_fig5` | `paper/figdata/fig5_mechanism.csv` | Verified |
| Fig. 6 accuracy/time Pareto (penalized primary) | `scripts/plots_nature.py::plot_fig6` | `paper/figdata/fig6_pareto.csv` | Verified |

---

## 5. Reliability, reproducibility and integrity statements

| Claim | Code | Data | Verdict |
|---|---|---|---|
| 78,605 ok / 2,977 timeout / 18 failed; ok fraction 96.3% | `scripts/audit_sweep_integrity.py` | `results.csv` (`status`) | Verified (96.330%) |
| 879 qEI rows exceeded the 56-GB soft guard; not a hardware benchmark | `scripts/check_sweep_runtime_health.py`, `tools/sanitize_results_rss.py` | `results.csv` (`rss_gb_peak`) | Verified (879 rows, all and only `botorch_qei`; max 226.06) |
| Per-output NRMSE: median spread 0.257, max 0.738; 21 cases > 0.1, 10 cases > 0.2 | `benchmark/scripts/generate_report.py` | `full_benchmark/analysis/per_output_nrmse.csv`; `paper/tables/si_per_output.tex` (S12) | Verified (0.257 / 0.738 / 21 / 10) |
| All 34 case sources agree with row-level and manifest fingerprints | `scripts/audit_sweep_integrity.py`, `benchmark/core.py::case_source_fingerprint` | `RUN_MANIFEST.json` `case_sources`, `results.csv` `case_source_sha256` | Verified (34/34 match; 20/20 method fingerprints match) |
| Every number/figure/table regenerates deterministically | `fill_paper_numbers.py`, `plots_nature.py`, `build_supplementary.py`, `prepare_figdata.py` | `numbers.json` `_provenance` (5 inputs, all SHA-256 match), `figure_manifest.json` (12 hashes all match) | Verified by re-execution |
| Case-library datasheet distributed with the archive | `scripts/generate_case_library_datasheet.py` | `docs/CASE_LIBRARY_DATASHEET.md` | **Added on 2026-09-14** (the file did not previously exist) |

---

## 6. What is NOT backed by an archived artifact

Honest negative list — nothing here is claimed as a generated result:

| Item | Status |
|---|---|
| Git tag `v0.4.0-prereg` (2026-05-01) | **Corrected 2026-09-14.** No tag exists and none was created: the development repository history began 2026-08-31, after the sweep, so no commit or tag can represent the pre-freeze state and tagging HEAD would misstate it. `v0.4.0-prereg` is a protocol version label, and `paper/preregistration.md` carries a freeze-provenance note stating that the freeze date is an author attestation. The label and compliance tests do not establish an independent pre-sweep timestamp. |
| `RUN_MANIFEST.json` run window | **Corrected 2026-09-14.** `run_window` now carries a `scope` note saying it describes the batch that produced the archived artefacts, and a separate `dataset_window` block records the true span 2026-06-11T16:39:05Z to 2026-06-17T23:36:50Z across three batches (67,080 + 2,514 + 12,006 = 81,600 rows, 34 cases). The Supplementary Information states the same. |
| `benchmark/results/provenance_repair_20260831/`, `reprovenance_rerun/` | Empty directories; the repair workflow that would populate them is documented but the intermediate outputs are not retained. |
| `_archive_20260831_d1_precanonical/` correlation (rho = 0.927) | Archived, superseded input. It was the origin of the stale `0.927` in the earlier `paper/numbers.tex`; the numerical artifacts were regenerated from the canonical archive on 2026-09-14 (rho = 0.914). The retained canonical artifacts, rather than the superseded input, underpin the evidence map above. |
