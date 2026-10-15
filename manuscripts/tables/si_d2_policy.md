# Supplementary Table S5 · D2 topology and denominator policy

The D2 headline analysis uses `penalized_failed_in_denominator`.
All 20 methods remain in each topology's denominator. Error medians
pool successful runs; a method without a qualifying median at any
tested budget is non-passing. Failed and timed-out runs are counted
separately, without a synthetic NRMSE penalty.

The D2 accuracy target is written ε (epsilon), with ε = 0.20 for the
registered analysis. The symbol τ (tau) is reserved for the D1 Kendall
rank correlation; the two are deliberately distinct.

Topologies with fewer than three registered cases are flagged
`exploratory=True` in `rule_summary.csv`. These rows are reported
separately and are not counted in the headline numerator or
denominator for the 10d rule.

The table S5 contract is the `rule_summary.csv` schema:
`topology`, `methods_total`, `methods_pass_at_10d`,
`frac_pass_at_10d`, `median_n_star`, `max_n_star`,
`n_methods_never_pass`, `n_cases`, `case_ids`, `exploratory`,
`n_failed_cells_per_topology`, `analysis_type` and `is_primary`.

