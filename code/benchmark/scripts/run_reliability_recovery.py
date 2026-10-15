"""
benchmark.scripts.run_reliability_recovery
==========================================
Discovery 3 of the Nature-grade benchmark: quantify the *signed*
failure-probability bias of each surrogate method on the 16 cases
that declare a ``failure_threshold``.

The script consumes an existing ``results.csv`` produced by
:mod:`benchmark.scripts.run_full_benchmark` (which already streams the
per-row reliability metrics through :mod:`benchmark.reliability`).
It does **not** re-run any benchmark — only post-hoc analysis.

Outputs (under ``--output-dir``)
--------------------------------
- ``reliability_long.csv``            — long-form per-run signed bias
- ``reliability_per_method.csv``      — primary cluster-bootstrap per-method summary
- ``reliability_per_method_iid.csv``  — i.i.d. bootstrap sensitivity summary
- ``reliability_ci_sensitivity.csv``  — cluster/i.i.d. CI-width ratios for SI
- ``reliability_ranking_abs_err.csv`` — Friedman-compatible ranking
- ``reliability_ranking_bal_acc.csv`` — same, balanced accuracy
- ``reliability_family_bias.csv``     — GP / RF / linear / BO grouping
- ``REPORT.md``                       — one-page narrative summary

Usage
-----
    python benchmark/scripts/run_reliability_recovery.py \
        benchmark/results/full_benchmark/results.csv \
        --budget-mult 10 \
        --output-dir benchmark/results/reliability_recovery
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

from benchmark.reliability_recovery import (
    per_method_reliability_summary,
    reliability_ci_sensitivity,
    reliability_recovery_report,
)
from scripts.prepare_figdata import (
    validate_headline_results_source,
    validate_results_contract,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Post-hoc Discovery-3 reliability recovery analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "results_csv",
        type=Path,
        help="Path to results.csv produced by run_full_benchmark.py.",
    )
    p.add_argument(
        "--budget-mult",
        type=int,
        default=None,
        help="If given, restrict the analysis to one budget; default = pool all.",
    )
    p.add_argument(
        "--output-aggregate",
        choices=["max", "mean"],
        default="max",
        help="How to aggregate multiple thresholded outputs into one bias.",
    )
    p.add_argument(
        "--cluster-by",
        type=str,
        default="case_id",
        choices=[None, "case_id", "method"],
        help=(
            "Column name to use as the cluster identifier for the "
            "bootstrap CI estimator.  Recommended: 'case_id' when the "
            "input has multiple repeats per case (otherwise the i.i.d. "
            "bootstrap underestimates standard errors). Default = case_id "
            "(pre-registered D3 primary estimator)."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "benchmark" / "results" / "reliability_recovery",
        help="Directory to write the analysis outputs.",
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
    if args.budget_mult is not None and int(args.budget_mult) <= 0:
        errors.append(f"--budget-mult must be positive when set; got {args.budget_mult}")
    if not str(args.cluster_by).strip():
        errors.append("--cluster-by must be a non-empty column name or omitted")
    return errors


def _format_summary_md(per_method: pd.DataFrame, family: pd.DataFrame,
                       budget_mult: Optional[int]) -> str:
    """Render the headline numbers as a Markdown report."""
    if per_method.empty:
        return "No reliability data found in the input CSV."

    best = per_method.iloc[0]
    worst = per_method.iloc[-1]
    most_neg_bias = per_method.loc[per_method["signed_bias_mean"].idxmin()]

    lines = [
        "# Reliability Recovery — Discovery 3",
        "",
        f"**Budget filter**: {'all budgets' if budget_mult is None else f'{budget_mult}d only'}  ",
        f"**Methods compared**: {len(per_method)}  ",
        f"**Total runs**: {int(per_method['n_runs'].sum())}",
        "",
        "## Headline numbers",
        "",
        f"- **Lowest |P_f| error**: `{best['method']}` "
        f"(mean abs error = {best['pf_abs_err_mean']:.3f}, "
        f"95 % CI [{best['pf_abs_err_ci_lo']:.3f}, {best['pf_abs_err_ci_hi']:.3f}])",
        f"- **Highest |P_f| error**: `{worst['method']}` "
        f"(mean abs error = {worst['pf_abs_err_mean']:.3f})",
        f"- **Least-conservative method (smallest signed bias)**: `{most_neg_bias['method']}` "
        f"(signed bias = {most_neg_bias['signed_bias_mean']:+.3f}, "
        f"Wilcoxon p(bias<0) = {most_neg_bias['signed_bias_pvalue_neg']:.3g}, "
        f"p(bias>0) = {most_neg_bias.get('signed_bias_pvalue_pos', float('nan')):.3g})",
        "",
        "## Family-level signed bias",
        "",
        family.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Method ranking by mean P_f abs-error",
        "",
        per_method[[
            "method", "method_family", "n_runs",
            "pf_abs_err_mean", "pf_abs_err_ci_lo", "pf_abs_err_ci_hi",
            "signed_bias_mean",
            "signed_bias_pvalue_neg", "signed_bias_pvalue_pos",
            "f_bal_acc_mean",
        ]].to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Interpretation",
        "",
        "- `signed_bias_mean > 0` → method *over*-estimates P_f "
        "(*conservative* in reliability terms).",
        "- `signed_bias_mean < 0` → method *under*-estimates P_f "
        "(*non-conservative*).",
        "- `pvalue_neg` tests H1: bias < 0 (non-conservative). "
        "p < 0.05 ⇒ significant non-conservatism.",
        "- `pvalue_pos` tests H1: bias > 0 (conservative). "
        "p < 0.05 ⇒ significant conservatism.",
        "- `pvalue_neg` ≈ 1 does *not* mean non-conservatism is "
        "rejected; it just means the data are far from the left tail. "
        "Check `pvalue_pos` for the opposite alternative.",
        "- Methods with high `f_bal_acc_mean` correctly classify failure "
        "vs. safe outcomes near the boundary even when their P_f estimate "
        "is biased.",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    """Run the registered D3 analysis: estimate surrogate-based probability of failure on the failure-boundary cases and write per-method signed-bias artifacts."""
    args = parse_args(argv)
    errors = _validate_args(args)
    if errors:
        for error in errors:
            print(f"[reliability_recovery:error] {error}", file=sys.stderr)
        return 2
    print(f"[reliability_recovery] reading {args.results_csv}")
    df = pd.read_csv(args.results_csv, low_memory=False)
    try:
        validate_results_contract(df)
        if args.require_headline_source:
            validate_headline_results_source(args.results_csv, df)
    except AssertionError as exc:
        print(f"[reliability_recovery:error] results.csv contract failed: {exc}", file=sys.stderr)
        return 2
    print(f"  shape = {df.shape}; status counts = "
          f"{df['status'].value_counts().to_dict() if 'status' in df.columns else 'n/a'}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = reliability_recovery_report(
        df,
        budget_mult=args.budget_mult,
        output_aggregate=args.output_aggregate,
        cluster_by=args.cluster_by,
    )

    if bundle["per_method"].empty:
        print("[reliability_recovery] no failure-probability columns "
              "in the input CSV — nothing to analyse.")
        return 1

    bundle["long"].to_csv(out_dir / "reliability_long.csv", index=False)
    bundle["per_method"].to_csv(out_dir / "reliability_per_method.csv", index=False)
    per_method_iid = per_method_reliability_summary(
        bundle["long"],
        budget_mult=args.budget_mult,
        cluster_by=None,
    )
    ci_sensitivity = reliability_ci_sensitivity(
        bundle["long"],
        budget_mult=args.budget_mult,
        primary_cluster_by=args.cluster_by or "case_id",
    )
    per_method_iid.to_csv(out_dir / "reliability_per_method_iid.csv", index=False)
    ci_sensitivity.to_csv(out_dir / "reliability_ci_sensitivity.csv", index=False)
    bundle["ranking_abs_err"].to_csv(out_dir / "reliability_ranking_abs_err.csv", index=False)
    bundle["ranking_bal_acc"].to_csv(out_dir / "reliability_ranking_bal_acc.csv", index=False)
    bundle["family_bias"].to_csv(out_dir / "reliability_family_bias.csv", index=False)

    md = _format_summary_md(bundle["per_method"], bundle["family_bias"],
                            budget_mult=args.budget_mult)
    (out_dir / "REPORT.md").write_text(md, encoding="utf-8")

    print(f"[reliability_recovery] outputs written to {out_dir}")
    print(f"  - reliability_long.csv          ({len(bundle['long'])} rows)")
    print(f"  - reliability_per_method.csv    ({len(bundle['per_method'])} methods)")
    print(f"  - reliability_per_method_iid.csv ({len(per_method_iid)} methods)")
    print(f"  - reliability_ci_sensitivity.csv ({len(ci_sensitivity)} methods)")
    print("  - reliability_ranking_abs_err.csv")
    print("  - reliability_ranking_bal_acc.csv")
    print(f"  - reliability_family_bias.csv   ({len(bundle['family_bias'])} families)")
    print("  - REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
