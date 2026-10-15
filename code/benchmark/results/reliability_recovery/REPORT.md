# Reliability Recovery — Discovery 3

**Budget filter**: 10d only  
**Methods compared**: 20  
**Total runs**: 9570

## Headline numbers

- **Lowest |P_f| error**: `botorch_saasbo` (mean abs error = 0.024, 95 % CI [0.010, 0.044])
- **Highest |P_f| error**: `random+linear` (mean abs error = 0.104)
- **Least-conservative method (smallest signed bias)**: `lhs+gp` (signed bias = +0.012, Wilcoxon p(bias<0) = 1, p(bias>0) = 1.42e-06)

## Family-level signed bias

| method_family              |   n_runs |   signed_bias_mean |   signed_bias_ci_lo |   signed_bias_ci_hi |   pvalue_one_sided_neg |   pvalue_one_sided_pos |
|:---------------------------|---------:|-------------------:|--------------------:|--------------------:|-----------------------:|-----------------------:|
| sequential_active_learning |      480 |              0.014 |              -0.002 |               0.033 |                  1.000 |                  0.000 |
| sequential_ego             |      480 |              0.014 |              -0.003 |               0.039 |                  1.000 |                  0.000 |
| sequential_botorch_saas    |      480 |              0.015 |              -0.001 |               0.038 |                  1.000 |                  0.000 |
| sequential_reliability     |      480 |              0.017 |              -0.002 |               0.044 |                  1.000 |                  0.000 |
| sequential_botorch_ei      |      450 |              0.022 |               0.001 |               0.052 |                  1.000 |                  0.000 |
| static_design              |     7200 |              0.030 |               0.003 |               0.066 |                  1.000 |                  0.000 |

## Method ranking by mean P_f abs-error

| method         | method_family              |   n_runs |   pf_abs_err_mean |   pf_abs_err_ci_lo |   pf_abs_err_ci_hi |   signed_bias_mean |   signed_bias_pvalue_neg |   signed_bias_pvalue_pos |   f_bal_acc_mean |
|:---------------|:---------------------------|---------:|------------------:|-------------------:|-------------------:|-------------------:|-------------------------:|-------------------------:|-----------------:|
| botorch_saasbo | sequential_botorch_saas    |      480 |             0.024 |              0.010 |              0.044 |              0.015 |                    1.000 |                    0.000 |            0.854 |
| halton+gp      | static_design              |      480 |             0.026 |              0.013 |              0.042 |              0.015 |                    1.000 |                    0.000 |            0.869 |
| lhs+gp         | static_design              |      480 |             0.026 |              0.013 |              0.043 |              0.012 |                    1.000 |                    0.000 |            0.863 |
| lhs_opt+gp     | static_design              |      480 |             0.027 |              0.013 |              0.046 |              0.019 |                    1.000 |                    0.000 |            0.859 |
| random+gp      | static_design              |      480 |             0.029 |              0.015 |              0.047 |              0.014 |                    1.000 |                    0.000 |            0.860 |
| ego_ei         | sequential_ego             |      480 |             0.029 |              0.013 |              0.050 |              0.014 |                    1.000 |                    0.000 |            0.845 |
| sobol+gp       | static_design              |      480 |             0.029 |              0.014 |              0.048 |              0.015 |                    1.000 |                    0.000 |            0.861 |
| max_variance   | sequential_active_learning |      480 |             0.030 |              0.016 |              0.047 |              0.014 |                    1.000 |                    0.000 |            0.891 |
| akmcs          | sequential_reliability     |      480 |             0.032 |              0.014 |              0.056 |              0.017 |                    1.000 |                    0.000 |            0.826 |
| botorch_qei    | sequential_botorch_ei      |      450 |             0.033 |              0.015 |              0.061 |              0.022 |                    1.000 |                    0.000 |            0.828 |
| lhs_opt+gp_rbf | static_design              |      480 |             0.040 |              0.021 |              0.063 |              0.017 |                    1.000 |                    0.000 |            0.810 |
| lhs_opt+rbf    | static_design              |      480 |             0.047 |              0.024 |              0.083 |              0.026 |                    1.000 |                    0.000 |            0.856 |
| lhs_opt+poly3  | static_design              |      480 |             0.049 |              0.025 |              0.086 |              0.017 |                    1.000 |                    0.000 |            0.824 |
| lhs_opt+poly5  | static_design              |      480 |             0.050 |              0.027 |              0.085 |              0.018 |                    1.000 |                    0.000 |            0.824 |
| lhs_opt+svr    | static_design              |      480 |             0.053 |              0.025 |              0.096 |              0.035 |                    1.000 |                    0.000 |            0.770 |
| lhs_opt+rf     | static_design              |      480 |             0.057 |              0.029 |              0.092 |              0.043 |                    1.000 |                    0.000 |            0.763 |
| lhs_opt+mlp    | static_design              |      480 |             0.062 |              0.028 |              0.107 |              0.044 |                    1.000 |                    0.000 |            0.825 |
| lhs_opt+knn    | static_design              |      480 |             0.073 |              0.040 |              0.114 |              0.058 |                    1.000 |                    0.000 |            0.722 |
| lhs_opt+gb     | static_design              |      480 |             0.103 |              0.059 |              0.155 |              0.056 |                    1.000 |                    0.000 |            0.704 |
| random+linear  | static_design              |      480 |             0.104 |              0.060 |              0.155 |              0.056 |                    1.000 |                    0.000 |            0.773 |

## Interpretation

- `signed_bias_mean > 0` → method *over*-estimates P_f (*conservative* in reliability terms).
- `signed_bias_mean < 0` → method *under*-estimates P_f (*non-conservative*).
- `pvalue_neg` tests H1: bias < 0 (non-conservative). p < 0.05 ⇒ significant non-conservatism.
- `pvalue_pos` tests H1: bias > 0 (conservative). p < 0.05 ⇒ significant conservatism.
- `pvalue_neg` ≈ 1 does *not* mean non-conservatism is rejected; it just means the data are far from the left tail. Check `pvalue_pos` for the opposite alternative.
- Methods with high `f_bal_acc_mean` correctly classify failure vs. safe outcomes near the boundary even when their P_f estimate is biased.