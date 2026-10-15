"""
benchmark.scripts.run_sample_complexity
========================================
Discovery 2 of the Nature-grade benchmark: per-topology sample
complexity ``n* / d`` for global accuracy.

Consumes an existing ``results.csv`` (no re-running of any benchmark)
and emits one CSV per analysis layer plus a markdown REPORT.

Usage
-----
    python benchmark/scripts/run_sample_complexity.py \
        benchmark/results/full_benchmark/results.csv \
        --threshold 0.20 \
        --output-dir benchmark/results/sample_complexity
"""

from __future__ import annotations

import argparse
import io
import sys
import warnings
from pathlib import Path
from typing import List, Optional

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

import pandas as pd

from benchmark.sample_complexity import sample_complexity_report
from scripts.prepare_figdata import (
    validate_headline_results_source,
    validate_results_contract,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=("Per-topology sample complexity (n*/d) analysis "
                     "for the DOE benchmark."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "results_csv",
        type=Path,
        help="Path to results.csv produced by run_full_benchmark.py.",
    )
    p.add_argument(
        "--metric",
        default="nrmse",
        help="Metric to study (must exist as a column in results.csv).",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.20,
        help=(
            "Target value of the metric (default 20 %% NRMSE).  "
            "Set to 0.05 for the ambitious engineering target (most "
            "method/topology pairs will fail it on the current benchmark)."
        ),
    )
    p.add_argument(
        "--rule-budget",
        type=int,
        default=10,
        help="Budget multiplier for the textbook rule of thumb.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "benchmark" / "results" / "sample_complexity",
        help="Directory to write outputs into.",
    )
    p.add_argument(
        "--require-headline-source",
        action="store_true",
        help=(
            "Fail unless results_csv is the canonical full_benchmark/results.csv "
            "and the CSV is the exact frozen 34 x 20 x 4 x 30 nature-pool grid. "
            "Use this in the formal paper pipeline."
        ),
    )
    return p.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    if not args.results_csv.exists():
        errors.append(f"results_csv does not exist: {args.results_csv}")
    if float(args.threshold) <= 0:
        errors.append(f"--threshold must be positive; got {args.threshold}")
    if int(args.rule_budget) <= 0:
        errors.append(f"--rule-budget must be positive; got {args.rule_budget}")
    if not str(args.metric).strip():
        errors.append("--metric must be non-empty")
    return errors


def _format_report_md(
    rule: pd.DataFrame,
    n_star: pd.DataFrame,
    *,
    threshold: float,
    rule_budget: int,
    metric: str,
) -> str:
    if rule.empty:
        return "No usable data found in the input CSV.\n"

    if "exploratory" in rule.columns:
        main_rule = rule[~rule["exploratory"].astype(bool)]
        exploratory_rule = rule[rule["exploratory"].astype(bool)]
    else:
        main_rule = rule
        exploratory_rule = rule.iloc[0:0]
    n_topo = len(main_rule)
    pass_at_rule = main_rule[main_rule["frac_pass_at_10d"] >= 0.5]
    headline = (
        f"At budget = {rule_budget}·d, {len(pass_at_rule)} of {n_topo} "
        f"main topology classes have ≥50 % of methods reaching the target "
        f"{threshold:.0%} {metric.upper()}."
    )

    lines: List[str] = [
        "# Sample-Complexity Recovery — Discovery 2",
        "",
        f"**Metric**: `{metric}`  ",
        f"**Threshold**: {threshold:.3f}  ",
        f"**Rule budget**: {rule_budget}·d (Loeppky et al. 2009)",
        "",
        "## Headline",
        "",
        headline,
        "",
        "## Per-topology summary",
        "",
        rule.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Exploratory topologies",
        "",
        (
            exploratory_rule.to_markdown(index=False, floatfmt=".3f")
            if len(exploratory_rule)
            else "None."
        ),
        "",
        "## n*/d table (rows=method, columns=topology)",
        "",
        "Lower is better.  ``NaN`` means *no observed budget* satisfied "
        "the threshold for that (method, topology) pair — typically "
        "indicating a fundamental sample-efficiency failure.",
        "",
        n_star.fillna("∞").round(2).to_markdown(),
        "",
        "## Interpretation",
        "",
        "- The 10·d rule of thumb (Loeppky 2009) is calibrated on "
        "analytical benchmarks.  Topologies with `frac_pass_at_10d` "
        "below 0.5 and median `n_star > 10` falsify the rule on "
        "physics-grounded engineering cases.",
        "- `n_methods_never_pass` quantifies how many methods are "
        "*structurally* unable to reach the accuracy target on this "
        "topology at any tested budget.",
        "- `n_failed_cells_per_topology` records failed or timed-out "
        "cells retained in the denominator audit.",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    """Run the registered D2 analysis: fit budget-vs-NRMSE curves per response topology and write the n* (smallest passing budget) artifacts."""
    args = parse_args(argv)
    errors = _validate_args(args)
    if errors:
        for error in errors:
            print(f"[sample_complexity:error] {error}", file=sys.stderr)
        return 2
    print(f"[sample_complexity] reading {args.results_csv}")
    df = pd.read_csv(args.results_csv, low_memory=False)
    try:
        validate_results_contract(df)
        if args.require_headline_source:
            validate_headline_results_source(args.results_csv, df)
    except AssertionError as exc:
        print(f"[sample_complexity:error] results.csv contract failed: {exc}", file=sys.stderr)
        return 2
    print(f"  shape = {df.shape}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = sample_complexity_report(
        df,
        metric=args.metric,
        nrmse_threshold=args.threshold,
        rule_budget=args.rule_budget,
    )

    bundle["learning"].to_csv(out_dir / "learning_curves.csv", index=False)
    bundle["n_star"].to_csv(out_dir / "n_star_per_topology.csv")
    bundle["rule_summary"].to_csv(out_dir / "rule_summary.csv", index=False)

    report = _format_report_md(
        bundle["rule_summary"], bundle["n_star"],
        threshold=args.threshold, rule_budget=args.rule_budget,
        metric=args.metric,
    )
    (out_dir / "REPORT.md").write_text(report, encoding="utf-8")

    print(f"[sample_complexity] outputs written to {out_dir}")
    print(f"  - learning_curves.csv      ({len(bundle['learning'])} rows)")
    print(f"  - n_star_per_topology.csv  (shape={bundle['n_star'].shape})")
    print(f"  - rule_summary.csv         ({len(bundle['rule_summary'])} topologies)")
    print("  - REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
