"""
benchmark.scripts.compare_runs
=================================
Diff two benchmark runs.  Useful for:

  - Comparing a pilot to a full benchmark and confirming the
    headline ranking is stable as N_cases grows.
  - Comparing before-and-after a methodological change (e.g. new
    surrogate, fixed bug, alternative metric).
  - Reproduction reports: ``--strict`` mode emits a non-zero exit
    code if any numeric column differs by more than a tolerance,
    suitable for the ``Benchmark / determinism`` CI job and for
    third-party reproduction submissions
    (``.github/ISSUE_TEMPLATE/benchmark_reproduction.yml``).

Reports
-------
1. Coverage diff (cases / methods / budgets / reps) between the two runs.
2. Mean-rank table on the intersection of cases and methods at the
   largest shared budget.
3. Spearman ρ between the two rank vectors → headline-stability score.
4. Per-method rank shift (run A → run B) sorted by absolute change.
5. Bootstrap CI overlap: are the two ranks statistically distinguishable?

Usage
-----
    python benchmark/scripts/compare_runs.py \
           --reference benchmark/results/_archive/full_benchmark_legacy_v01 \
           --candidate benchmark/results/full_benchmark

    # Strict numerical comparison (paired-seed determinism check):
    python benchmark/scripts/compare_runs.py \
           --reference benchmark/results/run_a \
           --candidate benchmark/results/run_b \
           --strict --tolerance 1e-9
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
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

import pandas as pd
from scipy.stats import spearmanr

from benchmark.analysis import critical_difference, friedman_test, pivot_metric, rank_methods
from benchmark.stratified import bootstrap_rank_ci


def _load(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p / "results.csv")
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    return df


def compare(a_dir: Path, b_dir: Path) -> None:
    a = _load(a_dir)
    b = _load(b_dir)

    print("=" * 72)
    print("Comparing benchmark runs:")
    print(f"  A = {a_dir}    ({len(a)} rows, {a['case_id'].nunique()} cases)")
    print(f"  B = {b_dir}    ({len(b)} rows, {b['case_id'].nunique()} cases)")
    print("=" * 72)

    # Intersection
    common_cases = sorted(set(a["case_id"]) & set(b["case_id"]))
    common_methods = sorted(set(a["method"]) & set(b["method"]))
    common_budgets = sorted(set(a["budget_mult"]) & set(b["budget_mult"]))

    print("\nIntersection:")
    print(f"  cases   : {len(common_cases)}: {common_cases}")
    print(f"  methods : {len(common_methods)}")
    print(f"  budgets : {sorted(int(x) for x in common_budgets)}")

    if not common_methods or not common_cases or not common_budgets:
        print("\n[error] no overlap between the two runs")
        return

    largest = max(common_budgets)

    # Filter both to the intersection
    a_f = a[(a["case_id"].isin(common_cases))
            & (a["method"].isin(common_methods))
            & (a["budget_mult"] == largest)]
    b_f = b[(b["case_id"].isin(common_cases))
            & (b["method"].isin(common_methods))
            & (b["budget_mult"] == largest)]

    # Pivot + rank
    pivot_a = pivot_metric(a_f, "nrmse")
    pivot_b = pivot_metric(b_f, "nrmse")

    ranks_a = rank_methods(pivot_a, lower_is_better=True)
    ranks_b = rank_methods(pivot_b, lower_is_better=True)

    # Align ordering
    methods = sorted(set(ranks_a.index) & set(ranks_b.index))
    ranks_a = ranks_a.reindex(methods)
    ranks_b = ranks_b.reindex(methods)

    # Headline diff table
    diff_df = pd.DataFrame({
        "method": methods,
        "rank_A": ranks_a.values,
        "rank_B": ranks_b.values,
    })
    diff_df["delta"] = diff_df["rank_B"] - diff_df["rank_A"]
    diff_df["abs_delta"] = diff_df["delta"].abs()
    diff_df = diff_df.sort_values("rank_B")
    print(f"\n=== Mean rank @ {largest}·d budget ({len(common_cases)} common cases, {len(methods)} common methods) ===")
    print(diff_df.round(3).to_string(index=False))

    # Friedman + CDs
    fa, pa = friedman_test(pivot_a)
    fb, pb = friedman_test(pivot_b)
    cda = critical_difference(len(methods), pivot_a.dropna().shape[0])
    cdb = critical_difference(len(methods), pivot_b.dropna().shape[0])

    print("\nFriedman test:")
    print(f"  A: chi^2 = {fa:.2f}, p = {pa:.2e}, CD = {cda:.3f}")
    print(f"  B: chi^2 = {fb:.2f}, p = {pb:.2e}, CD = {cdb:.3f}")
    if cdb < cda:
        print(f"  → B sharpens the resolution by {cda-cdb:.2f} mean-rank units (smaller CD).")
    else:
        print("  → A had sharper resolution.  Unusual; check coverage.")

    # Spearman rank correlation
    rho, p = spearmanr(ranks_a.values, ranks_b.values)
    print(f"\nSpearman ρ (rank-A vs rank-B) = {rho:.4f}  (p = {p:.2e})")
    if rho > 0.95:
        print("  → headline ranking is essentially preserved between runs.")
    elif rho > 0.8:
        print("  → strong but not perfect agreement; check the largest |delta| methods.")
    else:
        print("  → significant rank changes; the more comprehensive run revealed new ordering.")

    # Top-3 stability
    topk = 3
    top_a = ranks_a.sort_values().head(topk).index.tolist()
    top_b = ranks_b.sort_values().head(topk).index.tolist()
    print(f"\nTop-{topk} methods (A → B):")
    print(f"  A: {top_a}")
    print(f"  B: {top_b}")
    overlap = set(top_a) & set(top_b)
    print(f"  Top-{topk} overlap: {len(overlap)}/{topk}  ({sorted(overlap)})")

    # Bootstrap CI overlap on the largest movers
    print("\nBootstrap rank CI for the top movers:")
    movers = diff_df.sort_values("abs_delta", ascending=False).head(5)["method"].tolist()
    ci_a = bootstrap_rank_ci(a_f, budget_value=largest, n_boot=500, seed=0)
    ci_b = bootstrap_rank_ci(b_f, budget_value=largest, n_boot=500, seed=0)
    ci_a = ci_a.set_index("method")
    ci_b = ci_b.set_index("method")
    for m in movers:
        if m in ci_a.index and m in ci_b.index:
            la, ua = ci_a.loc[m, "ci_low"], ci_a.loc[m, "ci_high"]
            lb, ub = ci_b.loc[m, "ci_low"], ci_b.loc[m, "ci_high"]
            ma, mb = ci_a.loc[m, "mean_rank"], ci_b.loc[m, "mean_rank"]
            overlap_ci = (la <= ub) and (lb <= ua)
            tag = "overlap (no significant shift)" if overlap_ci else "no overlap (significant)"
            print(f"  {m:25s}  A: {ma:5.2f} [{la:5.2f}, {ua:5.2f}]   "
                  f"B: {mb:5.2f} [{lb:5.2f}, {ub:5.2f}]   {tag}")


def _strict_numeric_diff(a: pd.DataFrame, b: pd.DataFrame,
                         tolerance: float = 1e-9) -> int:
    """Return the number of (cell, column) pairs that differ by more than
    ``tolerance`` in absolute value, after aligning on the natural key
    ``(case_id, method, budget_mult, repeat)``.

    Used by the determinism CI gate: two runs with the same seed strategy
    must produce numerically identical numeric columns.
    """
    key = ["case_id", "method", "budget_mult", "repeat"]
    a_idx = a.set_index(key).sort_index()
    b_idx = b.set_index(key).sort_index()
    common_idx = a_idx.index.intersection(b_idx.index)
    a_idx = a_idx.loc[common_idx]
    b_idx = b_idx.loc[common_idx]
    print(f"\nStrict numeric comparison ({len(common_idx)} cells, tol={tolerance:g})")

    numeric_cols = sorted(set(a_idx.select_dtypes(include="number").columns)
                          & set(b_idx.select_dtypes(include="number").columns))
    n_diffs = 0
    max_abs = 0.0
    for col in numeric_cols:
        diff = (a_idx[col] - b_idx[col]).abs()
        bad = diff > tolerance
        n_bad = int(bad.sum())
        if n_bad:
            n_diffs += n_bad
            mx = float(diff.max())
            max_abs = max(max_abs, mx)
            print(f"  ✗ {col:30s}  {n_bad} cells differ; max |Δ| = {mx:.3e}")
    if n_diffs == 0:
        print(f"  ✓ all {len(numeric_cols)} numeric columns identical "
              f"to within tol={tolerance:g}")
    else:
        print(f"\n  TOTAL: {n_diffs} cells exceed tolerance; "
              f"global max |Δ| = {max_abs:.3e}")
    return n_diffs


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Diff two benchmark runs (pilot vs full, or A vs B).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--reference", type=Path, required=True,
                   help="Directory of the *reference* run (contains results.csv).")
    p.add_argument("--candidate", type=Path, required=True,
                   help="Directory of the *candidate* run to compare against.")
    p.add_argument("--strict", action="store_true",
                   help="Also perform a cell-by-cell numeric diff and exit "
                        "non-zero on any disagreement above --tolerance.")
    p.add_argument("--tolerance", type=float, default=1e-9,
                   help="Absolute tolerance for the strict numeric diff.")
    # Backwards-compat: accept positional A B if --reference omitted.
    p.add_argument("legacy_dirs", nargs="*", default=[],
                   help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    a_dir, b_dir = args.reference, args.candidate
    if not a_dir.is_dir() or not b_dir.is_dir():
        print(f"error: directories must exist (got {a_dir!s} and {b_dir!s})",
              file=sys.stderr)
        return 2
    compare(a_dir, b_dir)
    if args.strict:
        a = _load(a_dir)
        b = _load(b_dir)
        n_diffs = _strict_numeric_diff(a, b, tolerance=args.tolerance)
        return 0 if n_diffs == 0 else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
