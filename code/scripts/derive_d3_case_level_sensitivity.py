"""Derive a case-level sensitivity analysis for the registered D3 archive.

The registered D3 estimand is retained unchanged.  This companion analysis
first averages signed bias within each physical case at the 10d budget, then
uses the 16 case means as the independent units for a one-sided Wilcoxon test
and an equal-case bootstrap.  It is deliberately reported as post-hoc
sensitivity evidence, not as a replacement for the registered per-cell test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "benchmark" / "results" / "reliability_recovery" / "reliability_long.csv"
DEFAULT_JSON = ROOT / "benchmark" / "results" / "reliability_recovery" / "d3_case_level_sensitivity.json"
DEFAULT_CSV = ROOT / "benchmark" / "results" / "reliability_recovery" / "d3_case_level_sensitivity.csv"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive(input_path: Path, *, seed: int = 1000, bootstrap_b: int = 10_000) -> tuple[dict, list[dict]]:
    with input_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row for row in rows
        if row.get("status", "ok") == "ok"
        and int(row["budget_mult"]) == 10
        and row.get("pf_signed_bias", "") not in {"", "nan", "NaN"}
    ]
    by_case: dict[str, list[float]] = {}
    for row in selected:
        # The true probability grid is decimal with three places. Normalize
        # parsed values so pandas/CSV-reader binary tails cannot change ties.
        bias = float(np.round(float(row["pf_signed_bias"]), 12))
        by_case.setdefault(row["case_id"], []).append(bias)
    case_rows = [
        {
            "case_id": case_id,
            "n_runs": len(values),
            "mean_signed_bias": float(np.mean(values)),
            "median_signed_bias": float(np.median(values)),
        }
        for case_id, values in sorted(by_case.items())
    ]
    values = np.asarray([row["mean_signed_bias"] for row in case_rows], dtype=float)
    test = stats.wilcoxon(values, alternative="greater", zero_method="wilcox", method="auto")
    rng = np.random.default_rng(seed)
    bootstrap_means = np.asarray(
        [values[rng.integers(0, len(values), len(values))].mean() for _ in range(bootstrap_b)],
        dtype=float,
    )
    summary = {
        "schema_version": 1,
        "budget_mult": 10,
        "n_cases": int(len(values)),
        "n_runs": int(len(selected)),
        "n_positive_cases": int(np.sum(values > 0)),
        "n_negative_cases": int(np.sum(values < 0)),
        "n_zero_cases": int(np.sum(values == 0)),
        "equal_case_mean_signed_bias": float(values.mean()),
        "equal_case_bootstrap_ci_lo": float(np.quantile(bootstrap_means, 0.025)),
        "equal_case_bootstrap_ci_hi": float(np.quantile(bootstrap_means, 0.975)),
        "equal_case_bootstrap_B": int(bootstrap_b),
        "bootstrap_seed": int(seed),
        "wilcoxon_case_level_statistic": float(test.statistic),
        "wilcoxon_case_level_one_sided_greater_p": float(test.pvalue),
        "holm_family_threshold": 1.0 / 60.0,
        "holm_pass": bool(test.pvalue <= (1.0 / 60.0)),
        "source": input_path.relative_to(ROOT).as_posix(),
        "source_sha256": _sha256(input_path),
        "script": "scripts/derive_d3_case_level_sensitivity.py",
    }
    return summary, case_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--bootstrap-b", type=int, default=10_000)
    args = parser.parse_args()
    summary, case_rows = derive(args.input.resolve(), seed=args.seed, bootstrap_b=args.bootstrap_b)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "n_runs", "mean_signed_bias", "median_signed_bias"])
        writer.writeheader()
        writer.writerows(case_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
