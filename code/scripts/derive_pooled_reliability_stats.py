"""Derive the pooled reliability-bias and 10d-rule count-test statistics.

These four manuscript headline numbers are pooled across the whole D3/D2
populations and are therefore not produced by the per-method / per-topology
analysis stages.  This script recomputes them from the *frozen* sweep
artifacts so that every number quoted in the manuscript traces to a generated
file plus a runnable script (no hand-typed statistics).

Outputs
-------
benchmark/results/reliability_recovery/pooled_reliability_summary.json
benchmark/results/sample_complexity_0.20/tenx_rule_cp_bound.json

Run:  python scripts/derive_pooled_reliability_stats.py
"""
from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

import numpy as np
from scipy.stats import beta, wilcoxon

# Keep this script self-contained: callers may execute it from a parent
# workspace that contains an older ``benchmark`` package.  The value mirrors
# ``benchmark.reliability_recovery.PROBABILITY_DECIMAL_PLACES`` and documents
# the decimal precision used for the fixed 1,000-point probability grid.
PROBABILITY_DECIMAL_PLACES = 12

ROOT = Path(__file__).resolve().parents[1]
D3_DIR = ROOT / "benchmark" / "results" / "reliability_recovery"
D2_DIR = ROOT / "benchmark" / "results" / "sample_complexity_0.20"
PRIMARY_BUDGET_MULT = "10"
SEED = 1000          # matches sweep seed_start
N_BOOT = 10000


def pooled_d3(long_csv: Path) -> dict:
    """Pooled signed P_f bias at 10d with case-cluster bootstrap CI + Wilcoxon."""
    vals: list[float] = []
    by_case: dict[str, list[float]] = {}
    with long_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("budget_mult") != PRIMARY_BUDGET_MULT:
                continue
            try:
                v = round(float(row["pf_signed_bias"]), PROBABILITY_DECIMAL_PLACES)
            except (KeyError, ValueError):
                continue
            if math.isnan(v):
                continue
            vals.append(v)
            by_case.setdefault(row.get("case_id", "?"), []).append(v)

    n = len(vals)
    mean = sum(vals) / n
    median = float(np.median(vals))
    arr = np.asarray(vals)
    nz = arr[arr != 0.0]
    _, p_greater = wilcoxon(nz, alternative="greater", zero_method="wilcox", mode="approx")

    cases = list(by_case)
    rng = random.Random(SEED)
    boot = []
    for _ in range(N_BOOT):
        s = 0.0
        c = 0
        for _ in range(len(cases)):
            cv = by_case[cases[rng.randrange(len(cases))]]
            s += sum(cv)
            c += len(cv)
        boot.append(s / c)
    boot.sort()
    ci_lo = boot[int(0.025 * N_BOOT)]
    ci_hi = boot[int(0.975 * N_BOOT)]

    return {
        "budget_mult": int(PRIMARY_BUDGET_MULT),
        "n_runs": n,
        "n_cases": len(cases),
        "pooled_mean_signed_bias": mean,
        "pooled_median_signed_bias": median,
        "cluster_bootstrap_ci_lo": ci_lo,
        "cluster_bootstrap_ci_hi": ci_hi,
        "cluster_bootstrap_B": N_BOOT,
        "bootstrap_seed": SEED,
        "wilcoxon_one_sided_greater_p": float(p_greater),
        "_source": str(long_csv.relative_to(ROOT)),
        "_script": "scripts/derive_pooled_reliability_stats.py",
    }


def cp_upper_d2(rule_csv: Path) -> dict:
    """One-sided Clopper-Pearson 95% upper bound for the 10d pass-rate count test."""
    n_main = 0
    n_pass = 0
    with rule_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            explo = str(row.get("exploratory", "False")).strip().lower() in {"true", "1", "yes"}
            if explo:
                continue
            n_main += 1
            try:
                if float(row["frac_pass_at_10d"]) >= 0.5:
                    n_pass += 1
            except (KeyError, ValueError):
                pass
    # Clopper-Pearson one-sided upper bound: Beta(k+1, n-k) 0.95 quantile.
    cp_upper = float(beta.ppf(0.95, n_pass + 1, n_main - n_pass)) if n_pass < n_main else 1.0
    return {
        "n_main_topologies": n_main,
        "n_pass_at_10d": n_pass,
        "frac_pass": n_pass / n_main if n_main else float("nan"),
        "clopper_pearson_one_sided_upper95": cp_upper,
        "registered_cutoff_fraction": 4 / 7,
        "_source": str(rule_csv.relative_to(ROOT)),
        "_script": "scripts/derive_pooled_reliability_stats.py",
    }


def main() -> None:
    d3 = pooled_d3(D3_DIR / "reliability_long.csv")
    d2 = cp_upper_d2(D2_DIR / "rule_summary.csv")

    (D3_DIR / "pooled_reliability_summary.json").write_text(
        json.dumps(d3, indent=2), encoding="utf-8"
    )
    (D2_DIR / "tenx_rule_cp_bound.json").write_text(
        json.dumps(d2, indent=2), encoding="utf-8"
    )

    print("[D3 pooled]")
    print(f"  n_runs                = {d3['n_runs']}")
    print(f"  pooled signed bias    = {d3['pooled_mean_signed_bias']:.4f}")
    print(f"  pooled bias median    = {d3['pooled_median_signed_bias']:.4f}")
    print(f"  cluster bootstrap CI  = [{d3['cluster_bootstrap_ci_lo']:.5f}, {d3['cluster_bootstrap_ci_hi']:.4f}]")
    print(f"  Wilcoxon greater p    = {d3['wilcoxon_one_sided_greater_p']:.3e}")
    print("[D2 count test]")
    print(f"  pass {d2['n_pass_at_10d']}/{d2['n_main_topologies']}  CP upper95 = {d2['clopper_pearson_one_sided_upper95']:.4f}")


if __name__ == "__main__":
    main()
