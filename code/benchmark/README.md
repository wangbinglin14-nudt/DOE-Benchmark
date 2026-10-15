# `benchmark/` — DOE method benchmark suite

> **First-time readers**: start with [`../ARCHITECTURE.md`](../docs/overview/ARCHITECTURE.md)
> for the 5-second project overview and Stage 0–10 data flow, then
> come back here for the per-module walkthrough.  Other reference docs:
> [`../docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md) (how to
> run the sweep), [`../docs/STATISTICAL_ANALYSIS_PLAN.md`](../docs/STATISTICAL_ANALYSIS_PLAN.md)
> (what each Discovery tests), [`../docs/GLOSSARY.md`](../docs/GLOSSARY.md)
> (terminology).

Layered on top of `doe_testbed`, this subpackage adds:

- **Standardised evaluation protocol** (training/test split, fixed reference
  set, paired-sample seed derivation).
- **Up to 20 method adapters** across 5 samplers, 11 surrogates, 3 sequential
  active learners, and 2 BoTorch Bayesian-optimisation methods.  `nature_pool()`
  yields the 20-method ceiling; `full_method_pool()` yields the 18-method
  BoTorch-free subset.
- **Statistical rigour** — Friedman + Nemenyi post-hoc, Holm-Bonferroni
  corrected paired Wilcoxon, Cliff's δ with 95 % bootstrap CI, and bootstrap
  CIs on rank shifts between populations (e.g. analytical-vs-physical).
- **Three discovery analyses** (`ranking_reversal`, `sample_complexity`,
  `reliability_recovery`) each with a standalone CLI producing a tidy CSV
  and Markdown REPORT.
- **Analytical baselines** (Branin / Hartmann / Ishigami / Sobol G /
  Friedman / wing-weight) for the ranking-reversal experiment.
- **Publication-quality plots** with Demšar (2006) CD diagrams, Cliff's δ
  heatmaps, rank-CI forest plots, and stratified grids.

## Architecture (live, as of v0.4.0-dev)

```
benchmark/
├── __init__.py                  Public API (exports 40+ names)
├── core.py                      BenchmarkMethod / BenchmarkRunner /
│                                StaticDesignMethod + derive_seed
├── samplers.py                  Random / LHS / OptLHS / Sobol / Halton   (5)
├── surrogates.py                GP-Matern / GP-RBF / RBF / RF / GB /     (11)
│                                MLP / Poly / PolyChaos / SVR / KNN /
│                                Linear  (see SURROGATE_REGISTRY)
├── sequential.py                EGO/EI / Max-Variance / AK-MCS           (3)
├── bo_methods.py                BoTorch qEI / SAASBO                     (2)
├── methods.py                   default_ / extended_ / full_ / bo_ /
│                                nature_ pools                       (10/15/18/+2/20)
├── hyperparameters.py           Canonical HP table, pinned by CI
├── metrics.py                   RMSE / NRMSE / MAE / maxAE / q95AE /
│                                R² / NLL / PICP / MPIW
├── reliability.py               pf_abs_err / F1 / balanced-accuracy /
│                                Brier on per-case failure thresholds
├── importance.py                Sobol-ST recovery via Saltelli on a
│                                fitted surrogate
├── analysis.py                  Friedman / Nemenyi / rank tables /
│                                CD-diagram data
├── rigour.py                    Cliff's δ + bootstrap CI, pairwise
│                                win-rate, test-MC noise floor,
│                                per-output disaggregation, fit flags
├── stratified.py                per-VS / per-topology / per-domain /
│                                per-dim-bucket + bootstrap CI on
│                                ranks and rank shifts
├── multicriteria.py             Accuracy-vs-sensitivity / stability-vs-
│                                quality Pareto fronts
├── learning_curves.py           Power-law fits α(d), rank stability
├── dim_scaling.py               NRMSE vs dim, per-dim-bucket ranks
├── sensitivity_recovery.py      Discovery-2 support (Sobol recovery)
├── reliability_recovery.py      Discovery-3 (signed P_f bias,
│                                two-sided Wilcoxon)
├── sample_complexity.py         Discovery-2 (topology-stratified n*/d)
├── plots.py                     CD diagram, rank-CI forest,
│                                performance heatmap, Pareto, grids
├── analytical.py                Branin / Hartmann / Ishigami /
│                                Sobol-G / Friedman / wing-weight
├── scripts/                     (~30 CLI scripts, see below)
├── tests/                       281 unit/integration tests (see below)
└── results/                     Output directory (created on demand)
```

### CLI scripts

The most important scripts are listed below; `benchmark/scripts/` also
contains progress/timing/verification helpers.

| Script | What it does |
|---|---|
| `run_full_benchmark.py`          | Stage-2 sweep runner (argparse: `--pool`, `--budgets`, `--n-repeats`, `--seed-strategy`); formal Nature CS archives should enter through `scripts/run_pipeline.py` / `scripts/run_pipeline.sh`. |
| `run_ranking_reversal.py`        | Discovery 1: analytical vs physical ranks + bootstrap CI on shifts. |
| `run_sample_complexity.py`       | Discovery 2: per-topology n*/d with bootstrap CI. |
| `run_reliability_recovery.py`    | Discovery 3: signed P_f bias with two-sided Wilcoxon. |
| `run_sensitivity_recovery.py`    | Surrogate Sobol' recovery (Kendall τ vs ground truth). |
| `run_pilot.py`                   | Legacy pilot (8 × 10 × 3 × 5). |
| `run_extended_pilot.py`          | Extended pilot (12 × 15 × 2 × 4). |
| `analyze_full_benchmark.py`      | Legacy/post-hoc report + sensitivity helper; Nature CS headline artifacts should use `scripts/run_pipeline.py --profile full --from-stage 3 --confirm-full-run`. |
| `analyze_extended.py`            | Stratified analysis + bootstrap CI + plots. |
| `analyze_results.py`             | Lightweight Friedman / Nemenyi summary. |
| `generate_report.py`             | Generates a full Markdown REPORT.md (~1 000 lines). |
| `compare_runs.py`                | Diff two runs (+ `--strict` for cell-by-cell determinism). |
| `holm_demo.py`                   | Minimum reproducible example of the Holm procedure. |

### Test suite

Both `benchmark/tests/` and the repo-level `tests/` are exercised by CI;
see [`../tests/README.md`](../tests/README.md) for the two-tier split:

```bash
$ pytest tests benchmark/tests -q
...
~330 passed in ~2 min          # 2026-05-23
```

At 2026-05-23: **281 tests** in `benchmark/tests/` (smoke, sequential,
BO, extensions, reliability, stratified, hyperparameters, discoveries,
paper pipeline, pipeline orchestrator, doc-link integrity, palette
audit) and **~50 tests** in `tests/` (case-library: physics +
metadata + sampling utilities).

## Quick start

```bash
# Install benchmark dependencies (in addition to doe-testbed core)
pip install -r requirements.lock    # or: pip install -e ".[dev]"

# Verify infrastructure (2 min)
pytest benchmark/tests -q

# 2-minute smoke sweep (1 case × 3 methods × 2 budgets × 2 repeats)
python benchmark/scripts/run_full_benchmark.py \
    --pool default --cases cantilever_beam_5d \
    --budgets 10 --n-repeats 2 --n-test 200 \
    --output-dir benchmark/results/smoke

# Nature CS archive (local no-sweep handoff, then server only; use docs/SERVER_RUN_HANDOFF.md)
python scripts/check_local_pre_server_ready.py
python scripts/check_server_launch_ready.py --require-clean-sweep-dir
bash scripts/run_pipeline.sh
bash scripts/create_server_return_package.sh

# Discovery-1 analysis (needs a completed sweep + the analytical baseline)
python benchmark/scripts/run_ranking_reversal.py

# Discovery-3 analysis (post-hoc on results.csv)
python benchmark/scripts/run_reliability_recovery.py \
    benchmark/results/full_benchmark/results.csv --budget-mult 10

# Compare two runs (determinism or pilot-vs-full)
python benchmark/scripts/compare_runs.py \
    --reference benchmark/results/pilot_v03 --candidate benchmark/results/full_benchmark \
    --strict --tolerance 1e-9
```

## Programmatic API

```python
from benchmark import BenchmarkRunner, default_method_pool
from benchmark.analysis import summarise

runner = BenchmarkRunner(
    case_ids=["cantilever_beam_5d", "airfoil_stall_2d"],
    methods=default_method_pool(),
    budgets=[5, 10, 20],   # multiples of d
    n_repeats=10,
    n_test=2000,
    output_dir="my_results/",
)
result = runner.run()             # streams to results.csv

summary = summarise(result.df, metric="nrmse", budget_value=20)
print(summary["ranks"])           # mean rank per method
print(summary["win_count"])       # cases won per method
print(f"Friedman p = {summary['friedman_p']:.2e}")
```

## Method pools

Five named pools, increasing in size and compute.  The canonical
hyperparameter table for all methods lives in
`benchmark/hyperparameters.py` and is pinned against drift by
`benchmark/tests/test_hyperparameters.py`.

### `default_method_pool()` — 10 methods (legacy pilot)
| Sampler | Surrogate | Family |
|---|---|---|
| Random / LHS / Sobol / OptLHS / Halton | GP-Matern | sampler ablation × 5 |
| OptLHS | RBF / RF / MLP / Poly3 | surrogate ablation × 4 |
| Random | Linear | low-bias baseline |

### `extended_method_pool()` — 15 methods
`default_method_pool` + 5 additional surrogate ablations:
| Method | Notes |
|---|---|
| `lhs_opt+gp_rbf` | GP with squared-exponential kernel (kernel ablation) |
| `lhs_opt+svr` | Support Vector Regression (RBF kernel) |
| `lhs_opt+knn` | k-nearest-neighbours regression |
| `lhs_opt+gb` | Gradient Boosted Trees (HistGradientBoosting) |
| `lhs_opt+poly5` | Polynomial Chaos-style degree-5 ridge regression |

### `full_method_pool()` — 18 methods
`extended_method_pool` + 3 sequential / active-learning methods:
| Method | Family | Notes |
|---|---|---|
| `ego_ei` | sequential_ego | Bayesian optimisation with Expected Improvement |
| `max_variance` | sequential_active_learning | Greedy maximum-uncertainty |
| `akmcs` | sequential_reliability | Active Kriging + MCS |

### `bo_method_pool()` — 2 methods (BoTorch; optional)
Only returns non-empty if `botorch` is importable in the current Python:
| Method | Family | Notes |
|---|---|---|
| `botorch_qei` | bayesian_optimisation | qEI via `botorch.acquisition.qLogExpectedImprovement` |
| `botorch_saasbo` | bayesian_optimisation | SAAS prior + qEI (`pyro-ppl` + `botorch`) |

### `nature_pool()` — up to 20 methods
`full_method_pool()` + `bo_method_pool()` — the recommended pool for the
Nature-grade submission, deterministic fallback to 18 if BoTorch is not
installed.  The formal publication pipeline disables that fallback with
`--require-botorch --expected-methods 20` and also passes
`--require-formal-case-backends` so optional case simulators such as
NeuralFoil/AeroSandbox cannot silently change response surfaces.

## Metrics

| Metric    | Direction | Notes                                       |
|-----------|-----------|---------------------------------------------|
| `rmse`    | ↓         | Native units (output dimensional)           |
| `nrmse`   | ↓         | Scale-free; comparable across cases         |
| `mae`     | ↓         | Mean absolute error                          |
| `maxae`   | ↓         | Worst-case absolute error                    |
| `q95ae`   | ↓         | 95-th percentile absolute error              |
| `r2`      | ↑         | Coefficient of determination                 |
| `nll`     | ↓         | Gaussian negative log-likelihood (needs std) |
| `picp_95` | ↑(→ 0.95) | Prediction-interval coverage probability     |
| `mpiw_95` | ↓         | Mean prediction-interval width               |

## Comparison dimensions

The :mod:`stratified` module slices the full results table along five
orthogonal dimensions:

1. **VS scenario** (VS-A through VS-H) — DOE research applicability tag
2. **Response-surface topology** (smooth_monotone / smooth_unimodal /
   strong_interaction / heterogeneous / threshold_jump / sparse_effects /
   multi_modal / discontinuous / constrained_domain)
3. **Domain** (structures / aerodynamics / heat_transfer /
   chemical_process / multidisciplinary / geomechanics)
4. **Dimension bucket** (low: $d \le 4$; mid: $5 \le d \le 8$; high: $d \ge 9$)
5. **Cost level** (L0 / L1 / L2 / L3) — solver-class label

Bootstrap CIs are computed for both **mean ranks** (clustered bootstrap
over cases) and **rank shifts** between two case populations
(e.g. analytical vs physical).

## Reliability sub-metrics (16 cases with failure thresholds)

For each case with a stored ``failure_threshold`` the runner adds:

| Metric | Direction | Notes |
|---|---|---|
| `pf_true_o{j}` | — | Empirical $P_f$ on test set |
| `pf_pred_o{j}` | — | Surrogate-predicted $P_f$ |
| `pf_abs_err_o{j}` | ↓ | Absolute error on $P_f$ |
| `f_precision_o{j}` | ↑ | Failure detection precision |
| `f_recall_o{j}` | ↑ | Failure detection recall |
| `f_f1_o{j}` | ↑ | F1 score on failure indicator |
| `f_bal_acc_o{j}` | ↑ | Balanced accuracy |
| `f_brier_o{j}` | ↓ | Probabilistic Brier score (needs `Y_std`) |

## Variable-importance recovery

The :mod:`importance` module checks whether a fitted surrogate
recovers the case's stored Sobol total-effect indices via Saltelli
re-estimation:

| Metric | Direction | Notes |
|---|---|---|
| `kendall_tau_st` | ↑ | Rank correlation true vs surrogate $S_T$ |
| `spearman_rho_st` | ↑ | Same, Spearman variant |
| `mae_st` | ↓ | Absolute error on $S_T$ |
| `top_k_overlap` | ↑ | Fraction of true top-$\lceil d/2 \rceil$ recovered |

## Roadmap

| Phase | Scope                                                        | Status |
|-------|--------------------------------------------------------------|--------|
| 1     | Static design methods + standard metrics + pilot             | ✅ done |
| 2     | Analytical baselines + ranking-reversal experiment           | ✅ done |
| 3     | Sequential / active-learning methods (EGO, AK-MCS)           | ✅ done |
| 4     | Method pool extension (15 surrogates → 18 incl. sequential)  | ✅ done |
| 5     | Stratified analysis (5 dimensions) + bootstrap CIs           | ✅ done |
| 6     | Reliability metrics + Sobol-recovery metrics                 | ✅ done |
| 7     | Extended pilot (12 × 15 × 2 × 4)                             | ✅ done |
| 8     | Pilot full sweep (34 × 15 × 3 × 6 = 9 180 runs)              | ✅ done |
| 9     | BoTorch qEI + SAASBO adapters + `nature_pool`                | ✅ done |
| 10    | Canonical `hyperparameters.py` + CI drift check              | ✅ done |
| 11    | Holm-Bonferroni pairwise Wilcoxon + Cliff's δ CI             | ✅ done |
| 12    | Discovery pipeline (ranking-reversal, sample-complexity,     |       |
|       | reliability-recovery) with bootstrap CIs                     | ✅ done |
| 13    | Reproducibility infrastructure (Docker, Snakemake,           |       |
|       | requirements.lock, `compare_runs --strict` determinism CI)   | ✅ done |
| 14    | Nature-grade sweep (34 × 20 × 4 × 30 ≈ 80 k runs)            | ⏳ running |
| 15    | Manuscript submission (Nature Communications)                | ⏳ next |
| 16    | Multi-fidelity benchmark (Co-Kriging, NARGP)                 | ⏳ v0.5 |

## Citation

If you use this benchmark in academic work:

```bibtex
@software{wang_doe_benchmark_2026,
  author  = {Wang, Binglin},
  title   = {DOE-Benchmark: A Physics-Grounded Benchmark Suite for
             Design-of-Experiments Methods},
  year    = {2026},
  url     = {https://github.com/wangbinglin14/doe-testbed},
}
```
