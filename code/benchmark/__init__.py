"""
doe_benchmark — Benchmark suite for DOE/surrogate method evaluation
====================================================================

This subpackage layers a *benchmark protocol* on top of the
``doe_testbed`` test problem library.  Where ``doe_testbed`` provides
physics-grounded simulation cases under a unified interface, this
package provides:

- A standardised evaluation protocol (training/test split, metrics).
- A library of method adapters (samplers, surrogates, sequential
  strategies) with a single :class:`BenchmarkMethod` interface.
- An orchestrator (:class:`BenchmarkRunner`) that runs the cross
  product *cases × methods × budgets × repeats* and stores results
  in a tidy ``pandas.DataFrame``.
- Statistical analysis (Friedman/Nemenyi tests, critical-difference
  diagrams) for principled method comparison.

Quick start
-----------
>>> from benchmark import BenchmarkRunner, default_method_pool
>>> runner = BenchmarkRunner(
...     case_ids=["cantilever_beam_5d", "airfoil_stall_2d"],
...     methods=default_method_pool(),
...     budgets=[5, 10, 20],   # multiples of d
...     n_repeats=10,
... )
>>> results = runner.run()
>>> results.to_csv("pilot_results.csv")
"""

from benchmark.core import (
    BenchmarkMethod,
    BenchmarkResult,
    BenchmarkRunner,
    StaticDesignMethod,
    derive_seed,
)
from benchmark.dim_scaling import (
    alpha_vs_dim,
    default_dim_bands,
    nrmse_vs_dim,
    rank_by_dimension_band,
)
from benchmark.learning_curves import (
    fit_power_law,
    learning_curve_table,
    rank_stability_score,
    ranking_at_each_budget,
)
from benchmark.methods import (
    bo_method_pool,
    default_method_pool,
    extended_method_pool,
    full_method_pool,
    list_methods,
    nature_pool,
)
from benchmark.metrics import (
    METRIC_REGISTRY,
    compute_all_metrics,
    nrmse,
    rmse,
)
from benchmark.multicriteria import (
    accuracy_vs_sensitivity_pareto,
    compose_method_scorecard,
    stability_vs_quality_pareto,
)
from benchmark.reliability_recovery import (
    extract_long_signed_bias,
    per_method_reliability_summary,
    reliability_ranking,
    reliability_recovery_report,
    signed_bias_by_family,
)
from benchmark.rigour import (
    bootstrap_metric_ci,
    cliffs_delta,
    cliffs_delta_ci_matrix,
    cliffs_delta_matrix,
    dominance_score,
    failure_rate_per_method,
    fit_quality_flags,
    pairwise_signed_rank_test,
    pairwise_win_rate_matrix,
    pareto_front,
    per_output_nrmse,
    per_output_rank_shifts,
    per_output_summary,
)
from benchmark.sample_complexity import (
    compare_to_10d_rule,
    learning_curves_by_topology,
    n_star_per_topology,
    sample_complexity_report,
)
from benchmark.sensitivity_recovery import (
    compare_sensitivity,
    get_ground_truth_st,
    run_sensitivity_recovery,
    surrogate_response_range,
    surrogate_sobol_indices,
)
from benchmark.stratified import (
    annotate_results,
    bootstrap_rank_ci,
    bootstrap_rank_shift_ci,
    stratified_ranks,
    winner_table,
)

__all__ = [
    "BenchmarkMethod",
    "BenchmarkResult",
    "BenchmarkRunner",
    "StaticDesignMethod",
    "derive_seed",
    "default_method_pool",
    "extended_method_pool",
    "full_method_pool",
    "bo_method_pool",
    "nature_pool",
    "list_methods",
    "METRIC_REGISTRY",
    "compute_all_metrics",
    "rmse",
    "nrmse",
    # stratified analyses
    "annotate_results",
    "bootstrap_rank_ci",
    "bootstrap_rank_shift_ci",
    "stratified_ranks",
    "winner_table",
    # rigour diagnostics
    "bootstrap_metric_ci",
    "cliffs_delta",
    "cliffs_delta_matrix",
    "cliffs_delta_ci_matrix",
    "dominance_score",
    "failure_rate_per_method",
    "fit_quality_flags",
    "pairwise_signed_rank_test",
    "pairwise_win_rate_matrix",
    "pareto_front",
    "per_output_nrmse",
    "per_output_rank_shifts",
    "per_output_summary",
    # learning curves
    "fit_power_law",
    "learning_curve_table",
    "rank_stability_score",
    "ranking_at_each_budget",
    # sensitivity recovery
    "compare_sensitivity",
    "get_ground_truth_st",
    "run_sensitivity_recovery",
    "surrogate_response_range",
    "surrogate_sobol_indices",
    # reliability recovery (Discovery 3)
    "extract_long_signed_bias",
    "per_method_reliability_summary",
    "reliability_ranking",
    "reliability_recovery_report",
    "signed_bias_by_family",
    # sample complexity (Discovery 2)
    "compare_to_10d_rule",
    "learning_curves_by_topology",
    "n_star_per_topology",
    "sample_complexity_report",
    # dimensional scaling
    "alpha_vs_dim",
    "default_dim_bands",
    "nrmse_vs_dim",
    "rank_by_dimension_band",
    # multi-criteria
    "accuracy_vs_sensitivity_pareto",
    "compose_method_scorecard",
    "stability_vs_quality_pareto",
]

__version__ = "0.4.0.dev0"
