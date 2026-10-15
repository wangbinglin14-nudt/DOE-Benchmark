# Figure and table data traceability

Audit date: 2026-09-17. All plots are scripted visualizations of computational
results. No plotted values or scientific images were produced with generative
image tools. This audit checks retained files and regeneration; it cannot
independently authenticate the historical runs or validate models against
experiments that were not performed.

Workspace update (2026-09-23): the physical paper tree is now
`01_manuscripts/`; from `02_code/`, `paper/` points to that same tree. Its SI
fragments are under `paper/tables/`. The observations below describe the
2026-09-17 audit and are retained as a dated record.

## Fixed archive and figure chain

Headline CSV: `benchmark/results/full_benchmark/results.csv` (81,600 rows), SHA-256
`71e72d937b64abc6f5c44da2f6b2183e293e228258ce4c0abce9bb4100f0ca0b`.
There are 78,605 ok, 2,977 timeout and 18 failed cells; failures remain in the
archive. All 34 cases' row-level source fingerprints match the 30 shared Python
modules. Case evidence notes are additional documentation, not replacement code.

All figure CSVs are produced by `scripts/prepare_figdata.py`; all six figures
are drawn by `scripts/plots_nature.py`. The manuscript embeds
`paper/figures/pdf/fig1.pdf` through `fig4.pdf`; SI embeds `fig5.pdf` and `fig6.pdf`.
PDF, SVG, PNG and TIFF copies are exported from the same plotting call.

| Figure | Source data under `paper/figdata/` | Upstream source and meaning |
|---|---|---|
| 1 | `fig1_case_catalog.csv`, `fig1_method_catalog.csv` | Case registry; method names in run manifest and method-family labels in raw CSV. Catalog metadata, not experimental measurements. |
| 2 | `fig2_d1_ranks.csv`, `fig2_d1_spearman.csv` | `ranking_reversal/analytical/results.csv`, paired physical subsets, `pairing_*/rank_comparison.csv`, `correlation.json`; complete-case ranks and method-axis bootstrap intervals. |
| 3 | `fig3_d2_learning.csv`, `fig3_d2_rule.csv` | `sample_complexity_0.20/learning_curves.csv` and `rule_summary.csv`, regenerated with `run_sample_complexity.py` from raw results. Bands summarize variation; failed cells remain nonpassing denominator entries. |
| 4 | `fig4_d3_bias.csv` | `reliability_recovery/reliability_per_method.csv`, regenerated with `run_reliability_recovery.py --budget-mult 10`. Case-cluster confidence intervals and selected-output signed bias. |
| S1 | `fig5_mechanism.csv` | Raw CSV, registry topology and `benchmark.stratified.stratified_ranks` at 50d. Exploratory strata are marked. |
| S2 | `fig6_pareto.csv` | Raw NRMSE and runtime with `benchmark.rigour.pareto_front`; failures and successful-cell denominators are explicit. Runtime describes the archived host, not universal hardware performance. |

The raw D3 calculation rounds probability differences to 12 decimal places
before selecting the largest absolute difference. Exact ties select the first
output in column order. Ignoring this recorded rule can reverse the sign of a
numerically tied cell. The balanced-accuracy aggregate takes the maximum over
outputs, so it need not refer to the selected signed-bias output. `pf_true` is a
finite test-set reference, not an exact population probability.

## Supplementary table chain

Paths below are relative to `02_code/`. The `paper/` junction points to the single physical manuscript tree at `01_manuscripts/`; SI fragments are in `paper/tables/`. The `scripts/generate_si.py`
generator produces base fragments; `scripts/build_supplementary.py` prepares
the publication table layout; `scripts/review_response_analyses.py` generates
the additional robustness tables. The current manuscript tree contains every referenced fragment, so compilation does not require rerunning the sweep.

| Table | LaTeX fragment in `paper/tables/` | Input or generator |
|---|---|---|
| S1 | `si_hyperparameters_clean.tex` | `benchmark/hyperparameters.py`; `generate_si.py` → `si_hyperparameters.tex` → `build_supplementary.py` |
| S2 | `si_sobol_clean.tex` | `benchmark/data/` Sobol records; `generate_si.py` → `si_sobol.tex` → `build_supplementary.py` |
| S3 | `si_cases_clean.tex` | Registry, case-validation ledger, `fig1_case_catalog.csv`, base `si_cases.tex`; `build_supplementary.py` |
| S4 | `si_multiple_testing.tex` | Prespecified statistical policy; `generate_si.py` (policy text, not measured data) |
| S5 | `si_d2_policy.tex` | D2 rule-summary and denominator policy; `generate_si.py` |
| S6 | `si_d3_per_method.tex` | `reliability_per_method.csv`; `build_supplementary.py` |
| S7 | `si_d3_ci_sensitivity.tex` | `reliability_ci_sensitivity.csv`; `build_supplementary.py` |
| S8 | `si_d3_case_level.tex` | `d3_case_level_sensitivity.{csv,json}`; `derive_d3_case_level_sensitivity.py` then `build_supplementary.py` |
| S9 | `si_mc_noise.tex` | `full_benchmark/analysis/test_set_sufficiency.csv`; `build_supplementary.py`. Pooled-output error diagnostic from one fitted GP per audited case, not the primary output-averaged NRMSE. |
| S10 | `si_d1_loo.tex` | Archived analytical and physical D1 subsets; `review_response_analyses.py` |
| S11 | `si_d3_loco.tex` | `reliability_long.csv`, pooled summary; `review_response_analyses.py` |
| S12 | `si_per_output.tex` | Raw per-output NRMSE; `review_response_analyses.py` |
| S13 | `si_d1_tau.tex` | `ranking_reversal/correlation.json`; `review_response_analyses.py` |
| S14 | `si_design_model_guidance.tex` | Frozen `full_benchmark/results.csv`, case topology and method catalogs; `explore_design_model_guidance.py` (post-hoc descriptive case ranks) |

## Recheck results and limits

The 2026-09-17 read-only audit independently recomputed D1 ranks and correlations
for all six pairing/budget combinations and checked the physical rows against
the headline archive. D2 and D3 CSVs were regenerated from the raw archive using
the registered settings. Nine figure-source CSVs matched numerically, and all
six figure PDFs rendered identically at 144 dpi (zero changed pixel channels).
All scientific/protocol fields in `numbers.json` matched; generation date and
provenance timestamps are expected to differ. Thirty-one recorded checks passed.
The five software/data archives were also extracted into an isolated directory;
all 13 Supplementary Information table fragments were regenerated there and
matched the manuscript fragments exactly as text.

One reproducibility defect was repaired: the manifest-driven Figure 1 catalog
used sampler names as family labels. It now takes the family from the archive,
rejects conflicting per-method labels and reproduces the existing figure.
Two targeted regression tests cover this behavior. Raw data, case implementations
and the displayed scientific values were not changed.

Original case-source gaps and physical simplifications are catalogued in
[CASE_EVIDENCE_INDEX.md](CASE_EVIDENCE_INDEX.md). The data/code preparation follows
the [Nature Portfolio reporting policy](https://www.nature.com/nature-portfolio/editorial-policies/reporting-standards)
and [journal formatting guide](https://www.nature.com/commseng/submit/formatting-guidelines);
a local archive is not yet a public persistent deposit.
