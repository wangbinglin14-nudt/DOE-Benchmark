"""Health check for the headline 30-repeat full_benchmark sweep.

Pre-registration target (paper/preregistration.md §3, freeze v0.4.0-prereg):
    all registered cases x current nature_pool() x 4 budgets x 30 reps

Per-memory-class breakdown displayed in this report:
    light  = static_design methods
    medium = non-static methods except SAASBO
    heavy  = SAASBO, when available

Run from repo root:  python scripts/_check_sweep_progress.py
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import doe_testbed as db

CSV = Path("benchmark/results/full_benchmark/results.csv")

# ── Pre-registration target (paper/preregistration.md §3) ────────────────
PREREG_CASES = {case.case_id for case in db.list_cases()}
# Frozen nature_pool() contract.  Keeping this import-free lets read-only
# progress monitors run even on low-memory workstations before results.csv
# exists; server-launch gates and contract tests still compare it to the live
# registry so drift is caught before a formal run.
PREREG_METHOD_FAMILIES = {
    "random+gp": "static_design",
    "lhs+gp": "static_design",
    "sobol+gp": "static_design",
    "lhs_opt+gp": "static_design",
    "halton+gp": "static_design",
    "lhs_opt+rbf": "static_design",
    "lhs_opt+rf": "static_design",
    "lhs_opt+mlp": "static_design",
    "lhs_opt+poly3": "static_design",
    "random+linear": "static_design",
    "lhs_opt+gp_rbf": "static_design",
    "lhs_opt+svr": "static_design",
    "lhs_opt+knn": "static_design",
    "lhs_opt+gb": "static_design",
    "lhs_opt+poly5": "static_design",
    "ego_ei": "sequential_ego",
    "max_variance": "sequential_active_learning",
    "akmcs": "sequential_reliability",
    "botorch_qei": "sequential_botorch_ei",
    "botorch_saasbo": "sequential_botorch_saas",
}
LIGHT_METHODS = {
    name for name, family in PREREG_METHOD_FAMILIES.items() if family == "static_design"
}
HEAVY_METHODS = {
    name for name, family in PREREG_METHOD_FAMILIES.items() if "saas" in family.lower()
}
PREREG_METHODS = set(PREREG_METHOD_FAMILIES)
MEDIUM_METHODS = PREREG_METHODS - LIGHT_METHODS - HEAVY_METHODS
PREREG_BUDGETS = {5, 10, 20, 50}
PREREG_REPS = 30
PREREG_TARGET = (
    len(PREREG_CASES) * len(PREREG_METHODS) * len(PREREG_BUDGETS) * PREREG_REPS
)
GRID_COLUMNS = ["case_id", "method", "budget_mult", "repeat"]
VALID_STATUSES = {"ok", "timeout", "failed"}


def _protocol_mask(
    df: pd.DataFrame,
    cases: set[str],
    methods: set[str],
    budgets: set[int],
    repeats: set[int],
) -> pd.Series:
    """Rows that belong to the frozen target grid."""
    missing = [col for col in GRID_COLUMNS if col not in df.columns]
    if missing:
        return pd.Series(False, index=df.index)
    return (
        df["case_id"].isin(cases)
        & df["method"].isin(methods)
        & df["budget_mult"].isin(budgets)
        & df["repeat"].isin(repeats)
    )


def summarize_progress(
    df: pd.DataFrame,
    *,
    cases: set[str] | None = None,
    methods: set[str] | None = None,
    budgets: set[int] | None = None,
    n_repeats: int = PREREG_REPS,
) -> dict[str, object]:
    """Summarize progress by unique protocol cells, not raw CSV rows."""
    cases = cases or PREREG_CASES
    methods = methods or PREREG_METHODS
    budgets = budgets or PREREG_BUDGETS
    repeats = set(range(n_repeats))
    target = len(cases) * len(methods) * len(budgets) * n_repeats

    mask = _protocol_mask(df, cases, methods, budgets, repeats)
    protocol_rows = df[mask]
    unique_cells = protocol_rows.drop_duplicates(GRID_COLUMNS, keep="last")
    status_counts: dict[str, int] = {}
    invalid_statuses: list[str] = []
    if "status" in unique_cells.columns:
        raw_counts = unique_cells["status"].fillna("").astype(str).value_counts().to_dict()
        status_counts = {status: int(raw_counts.get(status, 0)) for status in sorted(VALID_STATUSES)}
        invalid_statuses = sorted(set(raw_counts) - VALID_STATUSES - {""})

    return {
        "rows_total": int(len(df)),
        "target_cells": int(target),
        "protocol_rows": int(len(protocol_rows)),
        "protocol_unique_cells": int(len(unique_cells)),
        "duplicate_protocol_rows": int(len(protocol_rows) - len(unique_cells)),
        "out_of_protocol_rows": int(len(df) - len(protocol_rows)),
        "status_counts_unique": status_counts,
        "invalid_statuses_unique": invalid_statuses,
    }


def _per_case_breakdown(df: pd.DataFrame, cases: set[str], methods: set[str]) -> pd.DataFrame:
    """Return per-case completion summary by unique protocol cells."""
    rows = []
    for case in sorted(cases):
        mask = _protocol_mask(
            df,
            {case},
            methods,
            PREREG_BUDGETS,
            set(range(PREREG_REPS)),
        )
        sub = df[mask].drop_duplicates(GRID_COLUMNS, keep="last")
        done = len(sub)
        target = len(methods) * len(PREREG_BUDGETS) * PREREG_REPS
        seen_methods = set(sub["method"].unique())
        missing = sorted(methods - seen_methods)
        status_counts = (
            sub["status"].fillna("").astype(str).value_counts().to_dict()
            if "status" in sub.columns
            else {}
        )
        rows.append(
            {
                "case": case,
                "done": done,
                "target": target,
                "pct": 100.0 * done / target if target else 0.0,
                "ok": int(status_counts.get("ok", 0)),
                "failed": int(status_counts.get("failed", 0)),
                "timeout": int(status_counts.get("timeout", 0)),
                "missing_methods": ", ".join(missing) if missing else "",
            }
        )
    return pd.DataFrame(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results_csv", type=Path, nargs="?", default=CSV)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    csv_path = args.results_csv
    if not csv_path.exists():
        print(f"results.csv not found at {csv_path}")
        return 0

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(
            f"could not parse results.csv at {csv_path}: "
            f"{type(exc).__name__}: {exc}"
        )
        print("Run scripts/check_sweep_runtime_health.py for the formal read-only health gate.")
        return 1
    mtime = datetime.fromtimestamp(os.path.getmtime(csv_path))
    age_min = (datetime.now() - mtime).total_seconds() / 60.0

    print("=" * 70)
    print("DOE-Benchmark sweep progress (full_benchmark)")
    print("=" * 70)
    print(f"CSV path        : {csv_path}")
    print(f"Total rows      : {len(df):,}")
    missing_grid = [col for col in GRID_COLUMNS if col not in df.columns]
    if df.empty:
        print("No data rows yet; waiting for the first streamed cell.")
        print("Run scripts/check_sweep_runtime_health.py --allow-missing for startup health.")
        return 0
    if missing_grid:
        print(f"Malformed results.csv: missing required progress columns {missing_grid}")
        print("Run scripts/check_sweep_runtime_health.py for the formal read-only health gate.")
        return 1
    print(f"Distinct cases  : {df['case_id'].nunique()} / {len(PREREG_CASES)}")
    print(f"Distinct methods: {df['method'].nunique()} / {len(PREREG_METHODS)}")
    extras = sorted(set(df["method"].dropna().astype(str).unique()) - PREREG_METHODS)
    if extras:
        print(f"Unexpected methods: {extras}")
    print(f"Last CSV update : {mtime}  ({age_min:.1f} min ago)")
    if age_min > 5:
        print(f"  WARNING: CSV not updated in {age_min:.1f} min - sweep may be dead.")
    print()

    # ── Pre-reg compliance summary ───────────────────────────────────────
    summary = summarize_progress(df)
    unique_cells = int(summary["protocol_unique_cells"])
    pct = 100.0 * unique_cells / max(PREREG_TARGET, 1)
    print(f"Pre-reg unique cells: {unique_cells:>6,} / {PREREG_TARGET:,}  ({pct:>5.1f}%)")
    print(f"Protocol rows       : {summary['protocol_rows']:>6,}")
    print(f"Duplicate protocol  : {summary['duplicate_protocol_rows']:>6,}")
    print(f"Out-of-protocol rows: {summary['out_of_protocol_rows']:>6,}")
    status_counts = summary["status_counts_unique"]
    if status_counts:
        print(
            "Unique-cell status : "
            f"ok={status_counts.get('ok', 0)} "
            f"failed={status_counts.get('failed', 0)} "
            f"timeout={status_counts.get('timeout', 0)}"
        )
    invalid_statuses = summary["invalid_statuses_unique"]
    if invalid_statuses:
        print(f"Invalid statuses    : {invalid_statuses}")

    # ── Per-memory-class breakdown ───────────────────────────────────────
    print()
    print("Per memory-class:")
    for cls_name, methods in [
        (f"light  ({len(LIGHT_METHODS)} static)".ljust(29), LIGHT_METHODS),
        (f"medium ({len(MEDIUM_METHODS)} seq/BO)".ljust(29), MEDIUM_METHODS),
        (f"heavy  ({len(HEAVY_METHODS)} SAASBO)".ljust(29), HEAVY_METHODS),
    ]:
        if not methods:
            continue
        target = len(PREREG_CASES) * len(methods) * len(PREREG_BUDGETS) * PREREG_REPS
        mask = _protocol_mask(df, PREREG_CASES, methods, PREREG_BUDGETS, set(range(PREREG_REPS)))
        sub = df[mask].drop_duplicates(GRID_COLUMNS, keep="last")
        done = len(sub)
        pct = 100.0 * done / max(target, 1)
        bar_len = int(pct / 5)
        bar = "#" * bar_len + "-" * (20 - bar_len)
        print(f"  {cls_name} [{bar}] {pct:>5.1f}%  ({done:>5}/{target:>5})")
    print()

    # ── Per-case breakdown ───────────────────────────────────────────────
    print("Per-case status (sorted by completion):")
    breakdown = _per_case_breakdown(df, PREREG_CASES, PREREG_METHODS)
    breakdown_sorted = breakdown.sort_values("pct", ascending=False)
    for _, row in breakdown_sorted.iterrows():
        bar_len = int(row["pct"] / 5)
        bar = "#" * bar_len + "-" * (20 - bar_len)
        status_suffix = ""
        if "ok" in row:
            status_suffix = f" ok={int(row['ok'])} failed={int(row['failed'])} timeout={int(row['timeout'])}"
        print(f"  {row['case']:<32} [{bar}] {row['pct']:>5.1f}%  "
              f"({row['done']:>4}/{row['target']}){status_suffix}")
    print()

    not_started = breakdown[breakdown["done"] == 0]
    if len(not_started) > 0:
        print(f"Not started yet ({len(not_started)} of {len(PREREG_CASES)} cases):")
        for case in not_started["case"]:
            print(f"  - {case}")
        print()

    # ── Suggested next action ────────────────────────────────────────────
    print("Suggestions:")
    if age_min > 10:
        print("  * Sweep appears stalled.  Inspect the server job/logs, then resume")
        print("    the formal frozen pipeline on Linux if the operator confirms it")
        print("    stopped:")
        print("    bash scripts/run_pipeline.sh")
        print("  * Do not launch the full sweep from a local review workstation.")
    elif unique_cells < PREREG_TARGET:
        print("  * Sweep in progress.  Watch live:")
        print("    Get-Content benchmark\\results\\full_benchmark\\run.log -Wait -Tail 30")
        print("    tail -f benchmark/results/full_benchmark/run.log    # Linux")
    else:
        print("  * Sweep complete.  Continue the frozen downstream pipeline:")
        print("    python scripts/run_pipeline.py --profile full --from-stage 3 --confirm-full-run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
