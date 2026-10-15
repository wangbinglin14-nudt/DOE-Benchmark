"""Derive the D3 robustness disclosures quoted in the manuscript.

The D3 headline reports a pooled signed bias with a case-cluster bootstrap
interval.  Three further quantities are needed so that the manuscript states
the *limits* of that headline rather than only the point estimate, and so that
none of them is hand-typed:

* how much of the pooled mean is contributed by its single most influential
  case, and what the case-cluster interval becomes when that case is removed;
* how many of the 20 methods have a cluster-bootstrap interval that excludes
  zero (a weaker bar than the registered per-cell Wilcoxon test); and
* how many methods survive a Holm correction when the 16 physical cases, not
  the per-cell rows, are treated as the independent units.

It also recomputes the reference scale used to express the pooled bias as a
percentage: the median true failure probability at the output that drives each
cell's signed bias, over the same D3 population.  This quantity previously had
no generated artifact behind it.

The bootstrap uses the identical recipe, seed and resample count as
``scripts/derive_pooled_reliability_stats.py`` so the full-sample interval
reproduced here matches the registered headline interval exactly.

Outputs
-------
benchmark/results/reliability_recovery/d3_sensitivity_extras.json

Run:  python scripts/derive_d3_sensitivity_extras.py
"""
from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

PROBABILITY_DECIMAL_PLACES = 12
PRIMARY_BUDGET_MULT = "10"
SEED = 1000          # matches sweep seed_start
N_BOOT = 10000

ROOT = Path(__file__).resolve().parents[1]
D3_DIR = ROOT / "benchmark" / "results" / "reliability_recovery"
RESULTS_CSV = ROOT / "benchmark" / "results" / "full_benchmark" / "results.csv"

PF_COLUMNS = [
    ("pf_true_o0", "pf_pred_o0"),
    ("pf_true_o1", "pf_pred_o1"),
    ("pf_true_o2", "pf_pred_o2"),
]


def _load_rows(long_csv: Path) -> list[tuple[str, str, float]]:
    rows: list[tuple[str, str, float]] = []
    with long_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("budget_mult") != PRIMARY_BUDGET_MULT:
                continue
            try:
                v = round(float(row["pf_signed_bias"]), PROBABILITY_DECIMAL_PLACES)
            except (KeyError, ValueError):
                continue
            if math.isnan(v):
                continue
            rows.append((row.get("case_id", "?"), row.get("method", "?"), v))
    return rows


def _cluster_bootstrap(
    by_case: dict[str, list[float]], cases: list[str]
) -> tuple[float, float]:
    """Percentile cluster bootstrap over cases (identical recipe to the headline)."""
    rng = random.Random(SEED)
    boot: list[float] = []
    for _ in range(N_BOOT):
        total = 0.0
        count = 0
        for _ in range(len(cases)):
            cv = by_case[cases[rng.randrange(len(cases))]]
            total += sum(cv)
            count += len(cv)
        boot.append(total / count)
    boot.sort()
    return boot[int(0.025 * N_BOOT)], boot[int(0.975 * N_BOOT)]


def _case_level_holm_survivors(
    rows: list[tuple[str, str, float]], cases: list[str]
) -> tuple[int, float]:
    """Count methods surviving Holm when cases are the independent units."""
    per_method: dict[str, dict[str, list[float]]] = {}
    for case_id, method, value in rows:
        per_method.setdefault(method, {}).setdefault(case_id, []).append(value)

    pvalues: list[tuple[str, float]] = []
    for method, by_case_method in per_method.items():
        means = np.array(
            [np.mean(by_case_method[c]) for c in cases if c in by_case_method]
        )
        if means.size < 2:
            continue
        try:
            _, p = wilcoxon(
                means, alternative="greater", zero_method="wilcox", mode="approx"
            )
        except ValueError:
            continue
        pvalues.append((method, float(p)))

    pvalues.sort(key=lambda item: item[1])
    k = len(pvalues)
    survivors = 0
    smallest_threshold = float("nan")
    for rank, (_, p) in enumerate(pvalues):
        threshold = 0.05 / (k - rank)
        if rank == 0:
            smallest_threshold = threshold
        if p <= threshold:
            survivors += 1
    return survivors, smallest_threshold


def _driving_output_reference_scale(
    results_csv: Path, cases: list[str], methods: list[str]
) -> dict:
    """Median true P_f at the output that drives each cell's signed bias.

    ``benchmark/reliability_recovery.py`` defines a row's signed bias as the
    ``pf_pred - pf_true`` difference at the output with the largest absolute
    discrepancy.  The median true P_f at that same output is the natural
    reference scale for reporting the pooled bias as a percentage.
    """
    import pandas as pd

    want = ["case_id", "method", "budget_mult", "status"]
    for true_col, pred_col in PF_COLUMNS:
        want += [true_col, pred_col]
    df = pd.read_csv(results_csv, usecols=want, low_memory=False)
    df = df[
        (df["budget_mult"] == int(PRIMARY_BUDGET_MULT))
        & (df["status"] == "ok")
        & (df["case_id"].isin(cases))
        & (df["method"].isin(methods))
    ].copy()

    pred = np.column_stack(
        [df[pred_col].to_numpy(dtype=float) for _, pred_col in PF_COLUMNS]
    )
    true = np.column_stack(
        [df[true_col].to_numpy(dtype=float) for true_col, _ in PF_COLUMNS]
    )
    bias = np.round(pred - true, PROBABILITY_DECIMAL_PLACES)
    usable = ~np.all(np.isnan(bias), axis=1)
    rows = np.where(usable)[0]
    idx = np.nanargmax(np.abs(bias[usable]), axis=1)
    return {
        "n_rows": int(rows.size),
        "median_true_pf_at_driving_output": float(np.median(true[rows, idx])),
    }


def d3_sensitivity_extras(long_csv: Path, per_method_csv: Path) -> dict:
    rows = _load_rows(long_csv)
    by_case: dict[str, list[float]] = {}
    for case_id, _, value in rows:
        by_case.setdefault(case_id, []).append(value)
    cases = list(by_case)
    n_runs = len(rows)

    pooled = sum(v for _, _, v in rows) / n_runs
    full_lo, full_hi = _cluster_bootstrap(by_case, cases)

    # Influence: share of the pooled mean contributed by each case.
    contributions = {c: sum(v) / n_runs for c, v in by_case.items()}
    top_case = max(contributions, key=lambda c: abs(contributions[c]))
    top_case_share_pct = 100.0 * contributions[top_case] / pooled

    reduced = [c for c in cases if c != top_case]
    loo_lo, loo_hi = _cluster_bootstrap(by_case, reduced)

    # Weaker bar than the registered per-cell test: does the interval cross zero?
    n_cluster_excludes_zero = 0
    n_methods_seen = 0
    with per_method_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            n_methods_seen += 1
            lo = float(row["signed_bias_ci_lo"])
            hi = float(row["signed_bias_ci_hi"])
            if lo > 0.0 or hi < 0.0:
                n_cluster_excludes_zero += 1
    n_cluster_includes_zero = n_methods_seen - n_cluster_excludes_zero

    n_case_holm, smallest_holm_threshold = _case_level_holm_survivors(rows, cases)

    scale = _driving_output_reference_scale(
        RESULTS_CSV, cases, sorted({m for _, m, _ in rows})
    )
    pooled_pct_of_reference = 100.0 * pooled / scale["median_true_pf_at_driving_output"]

    return {
        "budget_mult": int(PRIMARY_BUDGET_MULT),
        "n_runs": n_runs,
        "n_cases": len(cases),
        "pooled_mean_signed_bias": pooled,
        "cluster_bootstrap_ci_lo": full_lo,
        "cluster_bootstrap_ci_hi": full_hi,
        "most_influential_case": top_case,
        "most_influential_case_share_pct_of_pooled": top_case_share_pct,
        "loo_case_cluster_ci_lo": loo_lo,
        "loo_case_cluster_ci_hi": loo_hi,
        "n_methods_cluster_ci_excludes_zero": n_cluster_excludes_zero,
        "n_methods_cluster_ci_includes_zero": n_cluster_includes_zero,
        "n_methods_tested": n_methods_seen,
        "n_methods_case_level_holm_survivors": n_case_holm,
        "case_level_smallest_holm_threshold": smallest_holm_threshold,
        "n_rows_driving_output_scale": scale["n_rows"],
        "median_true_pf_at_driving_output": scale["median_true_pf_at_driving_output"],
        "pooled_bias_pct_of_median_reference_pf": pooled_pct_of_reference,
        "cluster_bootstrap_B": N_BOOT,
        "bootstrap_seed": SEED,
        "_source": str(long_csv.relative_to(ROOT)),
        "_secondary_source": str(RESULTS_CSV.relative_to(ROOT)),
        "_script": "scripts/derive_d3_sensitivity_extras.py",
    }


def main() -> None:
    extras = d3_sensitivity_extras(
        D3_DIR / "reliability_long.csv",
        D3_DIR / "reliability_per_method.csv",
    )
    out = D3_DIR / "d3_sensitivity_extras.json"
    out.write_text(json.dumps(extras, indent=2), encoding="utf-8")
    print(f"[ok] {out}")
    print(f"  most influential case   = {extras['most_influential_case']}"
          f" ({extras['most_influential_case_share_pct_of_pooled']:.1f}% of pooled)")
    print(f"  full cluster CI         = [{extras['cluster_bootstrap_ci_lo']:.5f},"
          f" {extras['cluster_bootstrap_ci_hi']:.5f}]")
    print(f"  drop-case cluster CI    = [{extras['loo_case_cluster_ci_lo']:.5f},"
          f" {extras['loo_case_cluster_ci_hi']:.5f}]")
    print(f"  methods whose CI excludes zero = "
          f"{extras['n_methods_cluster_ci_excludes_zero']}")
    print(f"  methods surviving case-level Holm = "
          f"{extras['n_methods_case_level_holm_survivors']}")
    print(f"  median true P_f at driving output = "
          f"{extras['median_true_pf_at_driving_output']:.4f}"
          f"  -> pooled bias is "
          f"{extras['pooled_bias_pct_of_median_reference_pf']:.2f}% of it")


if __name__ == "__main__":
    main()
