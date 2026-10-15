# Pre-registration decisions — DOE-Benchmark v0.4.0

> **Status**: RESULTS_FILLED / SIGNED_FOR_LOCAL_PACKAGE.
> This materialized decision ledger records the D1/D2/D3 interpretation read
> from the archived 30-repeat Nature-grade sweep outputs.  The local package
> intentionally skips the optional human case-validation attachment check, but
> the numerical discovery decisions below are pinned to the run manifest and
> generated analysis artifacts.

## Provenance

- **Run manifest**: `benchmark/results/full_benchmark/RUN_MANIFEST.json`
- **Manifest sha256 for results.csv**: `71e72d937b64abc6f5c44da2f6b2183e293e228258ce4c0abce9bb4100f0ca0b`
- **Sweep started (UTC)**: 2026-06-15T13:09:42.291996+00:00
- **Sweep finished (UTC)**: 2026-06-20T00:10:53.065781+00:00
- **Cluster / host**: ubuntu
- **Author signing this decision document**: Binglin Wang
- **Date signed (UTC+08:00)**: 2026-06-20T08:26:45+08:00

## D1 — H1: Spearman rho < 0.70 at 10d

Reference: SAP Section 2.1 and `paper/preregistration.md`.

| Pairing | Budget | rho | 95 percent CI | Holm-adjusted p |
|---|---:|---:|---|---:|
| dimension | 10d | 0.914 | [0.758, 0.971] | 0.999 threshold-test p; Holm fail |
| topology | 10d | 0.943 | [0.832, 0.982] | secondary pairing; CI upper > 0.70 |
| random | 10d | 0.935 | [0.806, 0.980] | secondary pairing; CI upper > 0.70 |

### Secondary (Kendall tau)

*Completed on 2026-09-14 under amendment A9: the registered sensitivity
statistic was archived but not transcribed when this ledger was signed.*

| Pairing | Budget | tau | 95 percent CI |
|---|---:|---:|---|
| dimension | 10d | 0.773 | [0.604, 0.902] |

The registered sensitivity statistic agrees with rho in sign and order at
the primary pairing.  Its interval is wider than the rho interval and its
lower bound lies below the 0.70 threshold, so it corroborates the direction
of the D1 headline rather than its full strength.  The complete
pairing--budget table, read from the same `correlation.json`, is
Supplementary Table S13.

- **Headline decision**: Falsified
- **Rationale**: The registered H1 alternative required Spearman rho below 0.70 at 10d.  The primary dimension pairing instead gave rho = 0.914 with a 95 percent CI of [0.758, 0.971].  Under the pre-registered claim-level falsification clause, D1 is therefore falsified because the upper bound is at least 0.70 and the two specified rank-shift intervals (random+gp and lhs_opt+rbf) both include zero.  Separately, the lower bound 0.758 is above 0.70, which supports the stronger leaderboard-level transfer interpretation reported in the manuscript; this is not a claim that the conventional null was formally accepted.
- **Robustness across pairings**: The topology and random pairings also remained high, with rho = 0.943 and rho = 0.935 respectively, so the completed archive supports rank transfer rather than rank reversal.
- **Manuscript impact**: Report D1 as analytical-to-physics rank transfer; do not claim the registered rank-reversal hypothesis was confirmed.

## D2 — H2: the 10d rule across topology classes

Reference: SAP Section 2.2 and `paper/preregistration.md`.

| Quantity | Value |
|---|---|
| pass_count / 7 | 3 / 7, point estimate 0.429 |
| Clopper-Pearson 95 percent CI | two-sided [0.099, 0.816]; one-sided upper 0.775 |
| Holm-adjusted p | 0.694 from exact binomial p = 0.347 |
| Exploratory topology bucket | 0 exploratory topologies; 7 primary topology classes |

- **Headline decision**: Inconclusive
- **Rationale**: The practical point estimate says the 10d rule failed in a majority sense, with only 3 of 7 topology classes passing.  However, the pre-registered one-sided Clopper-Pearson upper bound is 0.775, which is above the 4/7 = 0.571 rejection cutoff, so the strict confirmatory test does not confirm H2.
- **Manuscript impact**: Report the 10d rule as failing by point estimate and as underpowered/inconclusive under the exact pre-registered topology-count test.

## D3 — H3: probability-of-failure bias on failure cases

Reference: SAP Section 2.3 and `paper/preregistration.md`.

| Quantity | Value |
|---|---|
| Mean signed bias | 0.0264 at 10d |
| Median signed bias (per-cell location) | 0.0040 at 10d |
| 95 percent bootstrap CI | [0.001706, 0.059230] registered percentile case-cluster bootstrap |
| Wilcoxon one-sided p | 1.43e-251 |
| Holm-adjusted p | 4.30e-251 pooled; max method-level p_holm = 1.56e-4 |
| Direction | over-estimation of probability of failure |
| Effective n_runs | 9570 reliability runs across 16 case clusters |

- **Headline decision**: Confirmed
- **Rationale**: The 10d reliability runs show a positive mean signed bias (the reported effect size) with a case-cluster bootstrap CI excluding zero, while the registered Wilcoxon signed-rank test provides positive-location evidence with a one-sided p-value far below the Holm threshold. These are distinct estimands: the CI targets the mean and Wilcoxon targets distributional location. The method-level D3 table also reports 20 of 20 tested methods as Holm-significant, all in the over-estimation direction.
- **Manuscript impact**: Report D3 as confirmed positive probability-of-failure bias, with over-estimation interpreted as conservative in reliability terms.

## Holm-Bonferroni Ledger

| Discovery | Raw p | Rank | Holm threshold | Pass? |
|---|---:|---:|---:|---|
| D1 | 0.999 | 3 | 0.05 | N |
| D2 | 0.347 | 2 | 0.025 | N |
| D3 | 1.43e-251 | 1 | 0.01667 | Y |

## Falsification Log

### D1 falsification

SAP Section 6 clause: D1 is falsified if the upper 95 percent CI of rho at 10d
is at least 0.70.  The observed primary CI was [0.758, 0.971], so the
registered rank-reversal claim is falsified by the primary CI gate.

New manuscript headline: Analytical rankings transferred reliably to
physics-grounded cases; the registered rank-reversal hypothesis was not
confirmed.

### D2 falsification

SAP Section 6 clause: D2 is falsified if at least 4 topology classes pass at
m = 10 and NRMSE <= 0.20.  The observed pass count was 3 of 7, so the point
count does not falsify the practical rule-failure interpretation.  The stricter
Section 1 decision rule still requires the one-sided Clopper-Pearson upper
bound to be below 4/7; the observed upper bound was 0.775, so the confirmatory
test is inconclusive rather than confirmed.

New manuscript headline: The 10d rule failed by point estimate across topology
classes, but the registered exact topology-count test is underpowered and
inconclusive.

## Sign-off

Binglin Wang signs this local package decision ledger on
2026-06-20T08:26:45+08:00.  The numbers above are read directly from the
artefacts pinned in `RUN_MANIFEST.json`, including `results.csv`,
`correlation.json`, `rule_summary.csv`, `reliability_long.csv`, and
`reliability_per_method.csv`.  No manuscript text was amended before this
document was committed as the decision source for the results-filled editing
pass.
