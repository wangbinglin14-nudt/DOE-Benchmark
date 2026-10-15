"""Sweep progress breakdown for benchmark/results/full_benchmark.

Reports:
  1. Overall progress (rows / current pre-registered target).
  2. Iteration order: case -> method -> budget -> repeat.
  3. Per-method, per-budget mean wall-clock observed so far.
  4. Forward-projection of remaining time using per-(method, budget)
     timings on the current case as a baseline, scaled by case dim.

Note on extrapolation:
  - Static-design + EGO/MaxVar/AKMCS scale roughly linearly with d at
    fixed budget_mult (since n_train = budget_mult * d).
  - SAASBO is super-linear in d (NUTS posterior cost grows fast); its
    cost on aircraft_conceptual_8d (d=8) is a *lower bound* for the
    five 9d / 10d / 16d / 20d cases.

Usage:
  python scripts/sweep_progress_breakdown.py
  python scripts/sweep_progress_breakdown.py benchmark/results/full_benchmark/results.csv
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import doe_testbed as db
from benchmark.methods import nature_pool

CASE_IDS = sorted(c.case_id for c in db.list_cases())
METHODS = [m.name for m in nature_pool()]
BUDGETS = [5, 10, 20, 50]
N_REPEATS = 30
TARGET = len(CASE_IDS) * len(METHODS) * len(BUDGETS) * N_REPEATS
GRID_COLUMNS = ["case_id", "method", "budget_mult", "repeat"]
VALID_STATUSES = {"ok", "timeout", "failed"}
DEFAULT_RESULTS_CSV = ROOT / "benchmark" / "results" / "full_benchmark" / "results.csv"


def _health_gate_hint() -> str:
    return "Run scripts/check_sweep_runtime_health.py for the formal read-only health gate."


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results_csv", type=pathlib.Path, nargs="?", default=DEFAULT_RESULTS_CSV)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    p = args.results_csv.resolve()
    if not p.exists():
        print(f"results.csv not found at {p}")
        return 0
    if p.stat().st_size == 0:
        print(f"results.csv is empty at {p}")
        print(_health_gate_hint())
        return 1

    try:
        df = pd.read_csv(p)
    except Exception as exc:
        print(f"could not parse results.csv at {p}: {type(exc).__name__}: {exc}")
        print(_health_gate_hint())
        return 1

    required = set(GRID_COLUMNS + ["t_total_s"])
    missing = sorted(required - set(df.columns))
    if df.empty:
        print("No data rows yet; waiting for the first streamed cell.")
        print("Run scripts/check_sweep_runtime_health.py --allow-missing for startup health.")
        return 0
    if missing:
        print(f"Malformed results.csv: missing required progress columns {missing}")
        print(_health_gate_hint())
        return 1

    # ---- 1. Overall progress --------------------------------------------
    protocol_mask = (
        df["case_id"].isin(CASE_IDS)
        & df["method"].isin(METHODS)
        & df["budget_mult"].isin(BUDGETS)
        & df["repeat"].isin(range(N_REPEATS))
    )
    in_scope_rows = df[protocol_mask]
    in_scope = in_scope_rows.drop_duplicates(GRID_COLUMNS, keep="last")
    n_unique = len(in_scope)
    duplicate_protocol_rows = len(in_scope_rows) - n_unique
    out_of_protocol_rows = len(df) - len(in_scope_rows)
    extra_methods = sorted(set(df["method"].dropna().astype(str).unique()) - set(METHODS))
    print(f"Rows: {len(df):>6,}    "
          f"Unique cells: {n_unique:>6,}    "
          f"Target: {TARGET:>6,}    "
          f"Progress: {n_unique / TARGET:.2%}")
    print(f"Protocol rows: {len(in_scope_rows):>6,}    "
          f"Duplicates: {duplicate_protocol_rows:>6,}    "
          f"Out-of-protocol: {out_of_protocol_rows:>6,}")
    if "status" in in_scope.columns:
        status_counts = in_scope["status"].fillna("").astype(str).value_counts().to_dict()
        invalid_statuses = sorted(set(status_counts) - VALID_STATUSES - {""})
        print("Unique-cell status: "
              f"ok={int(status_counts.get('ok', 0))} "
              f"failed={int(status_counts.get('failed', 0))} "
              f"timeout={int(status_counts.get('timeout', 0))}")
        if invalid_statuses:
            print(f"Invalid statuses: {invalid_statuses}")
    if extra_methods:
        print(f"Unexpected methods outside nature_pool(): {extra_methods}")
    print()

    # ---- 2. Iteration order: which case is currently being processed ----
    if in_scope.empty:
        print("No protocol cells observed yet.")
        print(_health_gate_hint())
        return 1

    cases_seen = in_scope["case_id"].unique().tolist()
    current_case = in_scope["case_id"].iloc[-1]
    t_total_s = pd.to_numeric(in_scope["t_total_s"], errors="coerce").iloc[-1]
    t_display = f"{float(t_total_s):.1f}s" if pd.notna(t_total_s) else "<invalid>"
    print(f"Cases touched so far: {len(cases_seen)}")
    print(f"Current case        : {current_case}")
    print(f"Latest cell         : "
          f"{in_scope['method'].iloc[-1]} budget={in_scope['budget_mult'].iloc[-1]} "
          f"rep={in_scope['repeat'].iloc[-1]} t={t_display}")
    print()

    # ---- 3. Per-method, per-budget timing on the current case -----------
    print("=== Mean wall-clock per cell, by (method, budget_mult) on current case ===")
    print("    (NaN = not yet observed; iteration is method-major then budget-major)")
    cur = in_scope[in_scope["case_id"] == current_case].copy()
    cur["t_total_s"] = pd.to_numeric(cur["t_total_s"], errors="coerce")
    mb = cur.groupby(["method", "budget_mult"], observed=True)["t_total_s"].mean().unstack()
    for b in BUDGETS:
        if b not in mb.columns:
            mb[b] = float("nan")
    mb = mb[BUDGETS]
    print(mb.to_string(float_format=lambda x: f"{x:9.2f}"))
    print()

    # ---- 4. Project remaining wall-clock --------------------------------
    print("=== Remaining wall-clock projection (single-core, lower-bound) ===")
    cur_case_obj = db.get_case_meta(current_case)
    d_cur = cur_case_obj.dim_input
    all_cases = sorted(c.case_id for c in db.list_cases())
    dim_map = {c.case_id: c.dim_input for c in db.list_cases()}

    # Establish per-(method, budget) cost model from current case.
    cost_model = mb.ffill(axis=1).fillna(0.0).to_dict(orient="index")

    ALL_METHODS = METHODS

    remaining_s_total = 0.0
    remaining_per_method: dict = {m: 0.0 for m in ALL_METHODS}
    remaining_per_case: dict = {c: 0.0 for c in all_cases}

    for case_id in all_cases:
        d_other = dim_map[case_id]
        scale = d_other / d_cur  # static-design cost scales ~linearly in d
        for m in ALL_METHODS:
            for b in BUDGETS:
                done = len(in_scope[
                    (in_scope["case_id"] == case_id)
                    & (in_scope["method"] == m)
                    & (in_scope["budget_mult"] == b)
                ])
                remaining = N_REPEATS - done
                if remaining <= 0:
                    continue
                base = cost_model.get(m, {}).get(b, 0.0)
                if base == 0.0:  # method not yet observed on current case
                    base = mb[b].dropna().median() if mb[b].notna().any() else 60.0
                cell_cost = base * scale
                remaining_per_method[m] += remaining * cell_cost
                remaining_per_case[case_id] += remaining * cell_cost
                remaining_s_total += remaining * cell_cost

    print(f"Total remaining   : {remaining_s_total/3600:8.1f} h   "
          f"({remaining_s_total/86400:5.1f} days)")
    print()
    print("Top 5 methods by remaining cost:")
    for m, s in sorted(remaining_per_method.items(), key=lambda x: -x[1])[:5]:
        print(f"  {m:25s}  {s/3600:8.1f} h")
    print()
    print("Top 5 cases by remaining cost:")
    for c, s in sorted(remaining_per_case.items(), key=lambda x: -x[1])[:5]:
        print(f"  {c:30s}  d={dim_map[c]:>3d}  {s/3600:8.1f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
