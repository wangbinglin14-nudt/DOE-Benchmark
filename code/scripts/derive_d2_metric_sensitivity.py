"""Recompute the registered D2 10d-rule summary under an alternative metric.

Why this script exists
----------------------
The registered D2 headline ("the 10d lower-bound rule holds for 3 of 7
response-topology classes by point estimate") uses the three-output arithmetic
mean NRMSE (``nrmse``).  Because D2 is a *count* over topology classes, it is
sensitive to how the three outputs of a case are collapsed into one score: a
class that passes on the mean can fail as soon as the hardest output is used.

Rather than assert robustness, this script recomputes the whole registered
pipeline (same frozen archive, same 0.20 threshold, same 10d rule budget, same
penalised-failure denominator policy) with ``metric="nrmse_worst"`` and reports
both summaries side by side.  The ``nrmse`` recomputation is checked against the
archived ``rule_summary.csv`` so the comparison is anchored to the published
number rather than to a re-implementation.

Usage:

    python scripts/derive_d2_metric_sensitivity.py
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import pandas as pd

from benchmark.sample_complexity import PRIMARY_NRMSE_THRESHOLD, sample_complexity_report

RESULTS_CSV = ROOT / "benchmark" / "results" / "full_benchmark" / "results.csv"
D2_DIR = ROOT / "benchmark" / "results" / "sample_complexity_0.20"
RULE_BUDGET = 10
PRIMARY_METRIC = "nrmse"
ALTERNATE_METRIC = "nrmse_worst"


def _summarise(rule: pd.DataFrame, metric: str) -> dict:
    rule = rule.sort_values("topology").reset_index(drop=True)
    passing = rule[rule["frac_pass_at_10d"] >= 0.5]
    return {
        "metric": metric,
        "threshold": float(PRIMARY_NRMSE_THRESHOLD),
        "rule_budget": int(RULE_BUDGET),
        "n_classes": int(rule.shape[0]),
        "n_classes_point_estimate_pass": int(passing.shape[0]),
        "classes_point_estimate_pass": sorted(passing["topology"].tolist()),
        "per_topology": [
            {
                "topology": str(row["topology"]),
                "frac_pass_at_10d": float(row["frac_pass_at_10d"]),
                "methods_pass_at_10d": int(row["methods_pass_at_10d"]),
                "median_n_star": (
                    float(row["median_n_star"])
                    if pd.notna(row["median_n_star"]) else None
                ),
            }
            for _, row in rule.iterrows()
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-csv", type=Path, default=RESULTS_CSV)
    parser.add_argument("--out-dir", type=Path, default=D2_DIR)
    args = parser.parse_args(argv)

    if not args.results_csv.exists():
        print(f"[d2-metric-sensitivity:error] missing {args.results_csv}",
              file=sys.stderr)
        return 2

    df = pd.read_csv(args.results_csv, low_memory=False)
    if ALTERNATE_METRIC not in df.columns:
        print(f"[d2-metric-sensitivity:error] {ALTERNATE_METRIC!r} not in "
              f"{args.results_csv.name}", file=sys.stderr)
        return 2

    primary = sample_complexity_report(
        df, metric=PRIMARY_METRIC, nrmse_threshold=PRIMARY_NRMSE_THRESHOLD,
        rule_budget=RULE_BUDGET,
    )["rule_summary"]
    alternate = sample_complexity_report(
        df, metric=ALTERNATE_METRIC, nrmse_threshold=PRIMARY_NRMSE_THRESHOLD,
        rule_budget=RULE_BUDGET,
    )["rule_summary"]

    archived_path = args.out_dir / "rule_summary.csv"
    archived_matches = None
    if archived_path.exists():
        archived = pd.read_csv(archived_path)
        key = ["topology", "methods_pass_at_10d", "frac_pass_at_10d"]
        a = archived.sort_values("topology")[key].reset_index(drop=True)
        b = primary.sort_values("topology")[key].reset_index(drop=True)
        archived_matches = bool(a.equals(b))

    payload = {
        "source": str(args.results_csv.relative_to(ROOT)).replace("\\", "/"),
        "primary": _summarise(primary, PRIMARY_METRIC),
        "alternate": _summarise(alternate, ALTERNATE_METRIC),
        "archived_rule_summary_reproduced": archived_matches,
        "script": "scripts/derive_d2_metric_sensitivity.py",
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "d2_metric_sensitivity.json"
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    merged = primary[["topology", "frac_pass_at_10d", "methods_pass_at_10d"]].merge(
        alternate[["topology", "frac_pass_at_10d", "methods_pass_at_10d"]],
        on="topology", suffixes=("_nrmse", "_nrmse_worst"),
    ).sort_values("topology")
    merged.to_csv(args.out_dir / "d2_metric_sensitivity.csv", index=False)

    print(json.dumps(payload, indent=2))
    print(f"\nwrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
