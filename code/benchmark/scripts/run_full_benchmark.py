"""
benchmark.scripts.run_full_benchmark
======================================
Nature-grade Stage-2 sweep runner.

This script streams ``results.csv`` for a sweep.  The Nature CS headline
archive should be produced via ``scripts/run_pipeline.py`` / ``run_pipeline.sh``
so Stage 2b-10 integrity, analysis, figure, SI, manifest, and submission gates
run in order.  Direct invocations are useful for smoke/ad-hoc sweeps and for
server operators who need to resume Stage 2 explicitly.

Friedman / Nemenyi resolution
-----------------------------
The critical difference at a fixed number of methods scales as
``CD ∝ 1/√(N_cases · n_repeats)``.  The default configuration uses
``n_repeats = 30`` to match the resolution recommended by Demšar
(2006) for tens-of-method comparisons.  The publication pipeline selects
``--pool nature`` (20 methods); the script-level default remains ``full``
for backward-compatible ad hoc runs.

Method pools
------------
- ``--pool extended``  : 15 static-design methods (legacy default)
- ``--pool full``      : 18 methods incl. 3 sequential / active learners
- ``--pool nature``    : 20 pre-registered publication methods
- ``--pool default``   : 10 methods (pilot pool)

Usage
-----
Lightning quick smoke (only 1 case, default pool):

    python benchmark/scripts/run_full_benchmark.py \
        --cases cantilever_beam_5d --pool default \
        --budgets 5 10 --n-repeats 2

Standalone formal Stage-2 run (server/operator mode only):

    python benchmark/scripts/run_full_benchmark.py \
        --cases ALL --pool nature --budgets 5 10 20 50 --n-repeats 30 \
        --confirm-large-run --require-botorch \
        --output-dir benchmark/results/full_benchmark

Full publication archive (preferred):

    bash scripts/run_pipeline.sh
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import warnings
from pathlib import Path
from typing import List

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError:
    pass

import doe_testbed as db
from benchmark import (
    BenchmarkRunner,
    bo_method_pool,
    default_method_pool,
    extended_method_pool,
    full_method_pool,
    nature_pool,
)
from benchmark.methods import mf_pool, sota_pool

POOLS = {
    "default": default_method_pool,
    "extended": extended_method_pool,
    "full": full_method_pool,
    "bo": bo_method_pool,
    "nature": nature_pool,
    "sota": sota_pool,           # post-reg: nature + TURBO + D-optimal
    "mf": mf_pool,               # MF-only (1 method); pair with --cases-filter mf
}

LARGE_RUN_CELL_THRESHOLD = 10000


def _filter_cases(case_ids: List[str], filter_name: str) -> List[str]:
    """Apply a CaseMeta predicate to ``case_ids``.  See ``--cases-filter``."""
    if filter_name == "all":
        return case_ids

    def _meta(cid):
        return db.get_case(cid).meta

    if filter_name == "mf":
        return [c for c in case_ids if _meta(c).supports_multifidelity]
    if filter_name == "failure":
        return [c for c in case_ids if _meta(c).has_failure_boundary]
    if filter_name == "high_dim":
        return [c for c in case_ids if _meta(c).dim_input >= 8]
    if filter_name == "low_cost":
        return [c for c in case_ids if (_meta(c).cost_level or "L0") == "L0"]
    raise ValueError(f"unknown cases-filter {filter_name!r}")


def _resolve_case_ids(case_args: List[str] | None) -> List[str]:
    """Resolve CLI case arguments, accepting ``ALL`` as the full registry."""
    if not case_args or case_args == ["ALL"]:
        return sorted(c.case_id for c in db.list_cases())
    if "ALL" in case_args:
        raise SystemExit("--cases ALL must be used by itself; do not mix ALL with explicit case_id values")

    available = {c.case_id for c in db.list_cases()}
    requested = list(dict.fromkeys(case_args))
    unknown = sorted(set(requested) - available)
    if unknown:
        raise SystemExit(f"Unknown case_id(s) for --cases: {', '.join(unknown)}")
    return requested


def _validate_run_args(args: argparse.Namespace) -> list[str]:
    """Return direct-runner argument errors before any output is created."""
    errors: list[str] = []
    budgets = list(args.budgets or [])
    if not budgets:
        errors.append("--budgets must contain at least one positive integer")
    if any(int(b) <= 0 for b in budgets):
        errors.append(f"--budgets must be positive integers; got {budgets}")
    if len(set(int(b) for b in budgets)) != len(budgets):
        errors.append(f"--budgets must not contain duplicates; got {budgets}")
    if int(args.n_repeats) <= 0:
        errors.append(f"--n-repeats must be positive; got {args.n_repeats}")
    if int(args.n_test) <= 0:
        errors.append(f"--n-test must be positive; got {args.n_test}")
    if int(args.n_jobs) <= 0:
        errors.append(f"--n-jobs must be positive; got {args.n_jobs}")
    if args.bo_n_jobs is not None and int(args.bo_n_jobs) <= 0:
        errors.append(f"--bo-n-jobs must be positive when set; got {args.bo_n_jobs}")

    positive_float_options = [
        ("--timeout-per-cell", args.timeout_per_cell),
        ("--bo-timeout-per-cell", args.bo_timeout_per_cell),
        ("--ref-timeout-s", args.ref_timeout_s),
        ("--watchdog-s", args.watchdog_s),
        ("--max-wall-clock-h", args.max_wall_clock_h),
        ("--mem-cap-gb", args.mem_cap_gb),
    ]
    for label, value in positive_float_options:
        if value is not None and float(value) <= 0:
            errors.append(f"{label} must be positive when set; got {value}")
    if args.heartbeat_s is not None and float(args.heartbeat_s) < 0:
        errors.append(f"--heartbeat-s must be non-negative; got {args.heartbeat_s}")
    return errors


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse CLI options for the full 34-case x 20-method x 4-budget x 30-repeat sweep."""
    p = argparse.ArgumentParser(
        description="Run the full DOE benchmark sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--pool",
        choices=sorted(POOLS),
        default="full",
        help="Which method pool to use.",
    )
    p.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Case IDs to include; default or ALL = every case in doe_testbed.",
    )
    p.add_argument(
        "--cases-filter",
        choices=["all", "mf", "failure", "high_dim", "low_cost"],
        default="all",
        help=(
            "Predicate filter applied on top of --cases.  "
            "'mf'        = only cases with supports_multifidelity=True (5 cases).  "
            "'failure'   = only cases with has_failure_boundary=True  (16 cases).  "
            "'high_dim'  = only cases with d >= 8                      (7 cases).  "
            "'low_cost'  = only cases with cost_level == 'L0'.          "
            "Useful for the post-registration MF/SOTA robustness experiments."
        ),
    )
    p.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[5, 10, 20],
        help="Training budgets as multiples of input dimension d.",
    )
    p.add_argument(
        "--n-repeats",
        type=int,
        default=30,
        help="Number of independent repeats per (case, method, budget).",
    )
    p.add_argument(
        "--n-test",
        type=int,
        default=1000,
        help="Reference test set size.",
    )
    p.add_argument(
        "--seed-strategy",
        choices=["case_repeat", "legacy"],
        default="case_repeat",
        help=(
            "Per-cell seed derivation. 'case_repeat' (default) hashes "
            "(case_id, repeat) for true independence across cases; "
            "'legacy' replicates the v0.2 behaviour (seed_start + r)."
        ),
    )
    p.add_argument(
        "--seed-start",
        type=int,
        default=1000,
        help="Seed salt / starting offset.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "benchmark" / "results" / "full_benchmark",
        help="Directory to stream results.csv into.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Disable per-run progress messages.",
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help=(
            "Parallel workers (joblib loky backend).  1 = sequential. "
            "On an 8-logical-core machine, 6\u20138 yields near-linear "
            "speed-up for static methods; sequential / BO methods are "
            "limited by GIL-released BLAS so they parallelise well too. "
            "BLAS / OpenMP thread counts are pinned to 1 inside workers "
            "to avoid oversubscription."
        ),
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "If --output-dir/results.csv already contains rows for "
            "some (case, method, budget, repeat) cells, skip those "
            "cells instead of recomputing.  Use to resume an "
            "interrupted sweep or to top up after re-scoping."
        ),
    )
    p.add_argument(
        "--timeout-per-cell",
        type=float,
        default=None,
        help=(
            "Soft per-cell wall-clock cap (seconds) applied to every "
            "non-BoTorch method.  Cells that exceed this are recorded "
            "with status='timeout' and the runner moves on.  Default: "
            "no cap (legacy behaviour).  Recommended: 600s for a "
            "30-rep sweep so a single hung sklearn fit cannot stall "
            "the whole queue."
        ),
    )
    p.add_argument(
        "--bo-timeout-per-cell",
        type=float,
        default=None,
        help=(
            "Soft per-cell cap (seconds) for BoTorch methods "
            "(parallel_backend='threading').  These are typically far "
            "slower than scikit-learn methods (SAASBO at d=20 / "
            "budget=50 can run for >30 min), so they get a separate "
            "knob.  Default: falls back to --timeout-per-cell."
        ),
    )
    p.add_argument(
        "--ref-timeout-s",
        type=float,
        default=600.0,
        help=(
            "Cap on reference-set construction (case.sample + "
            "case.evaluate) per case.  Engineering / FEM cases that "
            "hang in their solver are skipped with a warning instead "
            "of freezing the entire sweep.  Default 600 s."
        ),
    )
    p.add_argument(
        "--watchdog-s",
        type=float,
        default=None,
        help=(
            "If joblib's parallel generator goes silent for this many "
            "seconds the partition is aborted and the remaining tasks "
            "fall back to serial.  Default: 2 * max(per-cell timeout) "
            "+ 60 s; set explicitly to override."
        ),
    )
    p.add_argument(
        "--heartbeat-s",
        type=float,
        default=60.0,
        help=(
            "How often (seconds) to emit a one-line liveness signal "
            "while the sweep is running.  Set to 0 to disable.  "
            "Useful so stdout never goes longer than this without "
            "showing progress + ETA, even during long BoTorch / "
            "SAASBO cells."
        ),
    )
    p.add_argument(
        "--max-wall-clock-h",
        type=float,
        default=None,
        help=(
            "Total wall-clock budget for this invocation, in hours.  "
            "Once exceeded the runner stops dispatching new "
            "(case, partition) blocks and returns whatever has been "
            "streamed.  Re-running with --skip-existing will resume "
            "from where it left off.  Default: no cap."
        ),
    )
    p.add_argument(
        "--confirm-large-run",
        action="store_true",
        help=(
            "Required when the planned grid has at least "
            f"{LARGE_RUN_CELL_THRESHOLD:,} cells.  This prevents accidental "
            "local launches of publication-scale sweeps while leaving smoke "
            "and targeted single-test runs easy to execute."
        ),
    )
    p.add_argument(
        "--require-botorch",
        action="store_true",
        help=(
            "For formal publication nature-pool runs, require the complete "
            "BoTorch/GPyTorch/Torch stack before dispatch so the 20-method "
            "headline protocol cannot silently degrade to 18 methods."
        ),
    )
    p.add_argument(
        "--require-formal-case-backends",
        action="store_true",
        help=(
            "Require optional simulator packages used by formal case response "
            "surfaces before dispatch.  Use with --pool nature for publication "
            "runs so case backends cannot silently change."
        ),
    )
    p.add_argument(
        "--require-runtime-monitoring",
        action="store_true",
        help=(
            "Require psutil before dispatch so formal server runs record RSS "
            "traces and enforce --mem-cap-gb instead of silently losing "
            "resource-monitoring evidence."
        ),
    )
    p.add_argument(
        "--bo-n-jobs",
        type=int,
        default=None,
        help=(
            "Worker count override for methods declaring "
            "memory_class='heavy' (typically BoTorch SAASBO).  Default "
            "is max(1, n_jobs // 4) which keeps RSS under ~n_jobs * 1 GB "
            "on a 16 GB host.  Set to 1 for the most defensive Windows "
            "configuration; raise to 8 or 16 on RAM-rich Linux servers."
        ),
    )
    p.add_argument(
        "--mem-cap-gb",
        type=float,
        default=None,
        help=(
            "Soft RSS cap (GB).  When the parent process exceeds this "
            "the runner downgrades the next case to serial dispatch and "
            "runs gc.collect().  Requires the optional `psutil` package; "
            "without it the cap is silently ignored.  Recommended on "
            "16 GB Windows hosts: --mem-cap-gb 12 (leaves 4 GB for OS "
            "+ IDE)."
        ),
    )
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    """Launch the registered full benchmark sweep: resolve the method pool and case list, validate resume state, then dispatch every cell to the parallel runner."""
    args = parse_args(argv)
    arg_errors = _validate_run_args(args)
    if arg_errors:
        for error in arg_errors:
            print(f"[error] {error}", file=sys.stderr)
        return 2

    case_ids = _resolve_case_ids(args.cases)

    if (
        args.require_botorch
        or args.require_formal_case_backends
        or args.require_runtime_monitoring
    ):
        from scripts.check_optional_dependencies import check_pool
        expected_methods = 20 if args.pool == "nature" else None
        rc, messages = check_pool(
            args.pool,
            require_botorch=bool(args.require_botorch),
            expected_methods=expected_methods,
            require_formal_case_backends=bool(args.require_formal_case_backends),
            require_runtime_monitoring=bool(args.require_runtime_monitoring),
        )
        if rc != 0:
            for msg in messages:
                print(f"[dependency] {msg}", file=sys.stderr)
            return 2

    # Apply CaseMeta predicate filter (used by post-registration MF / SOTA experiments)
    if args.cases_filter != "all":
        before = len(case_ids)
        case_ids = _filter_cases(case_ids, args.cases_filter)
        if not case_ids:
            raise SystemExit(
                f"--cases-filter {args.cases_filter!r} produced an empty case list "
                f"(originally {before} cases).  Aborting."
            )
        print(
            f"[--cases-filter {args.cases_filter}]  {before} -> {len(case_ids)} cases: "
            f"{', '.join(case_ids)}"
        )

    methods = POOLS[args.pool]()
    budgets = list(args.budgets)
    out_dir = Path(args.output_dir)

    n_total = len(case_ids) * len(methods) * len(budgets) * args.n_repeats
    if n_total >= LARGE_RUN_CELL_THRESHOLD and not args.confirm_large_run:
        print(
            "[error] refusing to start a large benchmark grid "
            f"({n_total:,} cells) without --confirm-large-run. "
            "Use smoke-sized settings for local checks, or run the confirmed "
            "large sweep on the designated server.",
            file=sys.stderr,
        )
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    # Sequential methods take ~10× longer; estimate generously.
    has_sequential = any(m.family != "static_design" for m in methods)
    per_run_s = 6.0 if has_sequential else 1.6
    parallel_speedup = max(1, args.n_jobs)
    print("=" * 72)
    print("DOE-Benchmark Full Run")
    print("=" * 72)
    all_cases_selected = (not args.cases) or args.cases == ["ALL"]
    print(f"Cases     ({len(case_ids):2d}): {'all of doe_testbed' if all_cases_selected else ', '.join(case_ids)}")
    print(f"Methods   ({len(methods):2d}): {', '.join(m.name for m in methods)}")
    print(f"Pool       : {args.pool}")
    print(f"Budgets    : {budgets}  (multipliers of d)")
    print(f"Repeats    : {args.n_repeats}")
    print(f"Test size  : {args.n_test}")
    print(f"Seed       : strategy={args.seed_strategy}, salt={args.seed_start}")
    print(f"n_jobs     : {args.n_jobs}")
    print(f"skip_exist : {args.skip_existing}")
    print(f"timeout    : per-cell={args.timeout_per_cell}s, "
          f"bo-cell={args.bo_timeout_per_cell or args.timeout_per_cell}s, "
          f"ref={args.ref_timeout_s}s")
    print(f"watchdog   : {args.watchdog_s}s (auto if None)")
    print(f"heartbeat  : {args.heartbeat_s}s (0 disables)")
    print(f"max-wall   : {args.max_wall_clock_h}h (None = no cap)")
    print(f"bo-n-jobs  : {args.bo_n_jobs} (None = auto max(1, n_jobs//4))")
    print(f"mem-cap    : {args.mem_cap_gb}GB (None = no soft cap)")
    print(f"Total runs : {n_total}")
    print(f"Wall-clock ≊ {n_total * per_run_s / parallel_speedup / 60:.0f} min  "
          f"({n_total * per_run_s / parallel_speedup / 3600:.1f} h, "
          f"\u00d7{parallel_speedup} parallel)")
    print(f"Output     : {out_dir}")
    print("=" * 72)

    t0 = time.perf_counter()
    runner = BenchmarkRunner(
        case_ids=case_ids,
        methods=methods,
        budgets=budgets,
        n_repeats=args.n_repeats,
        n_test=args.n_test,
        seed_start=args.seed_start,
        seed_strategy=args.seed_strategy,
        output_dir=out_dir,
        verbose=not args.quiet,
        n_jobs=args.n_jobs,
        skip_existing=args.skip_existing,
        timeout_s=args.timeout_per_cell,
        bo_timeout_s=args.bo_timeout_per_cell,
        ref_timeout_s=args.ref_timeout_s,
        watchdog_s=args.watchdog_s,
        heartbeat_s=args.heartbeat_s,
        max_wall_clock_s=(
            args.max_wall_clock_h * 3600.0 if args.max_wall_clock_h else None
        ),
        bo_n_jobs=args.bo_n_jobs,
        mem_cap_gb=args.mem_cap_gb,
    )
    result = runner.run()
    elapsed = (time.perf_counter() - t0) / 60
    print(f"\n=== Full benchmark completed in {elapsed:.1f} min ===")

    # Streamed CSV is the canonical artifact (incremental + crash-safe).
    # When --skip-existing is set, the in-memory `result` only holds the
    # newly-computed rows, while the streamed file holds *all* rows;
    # overwriting from `result.df` would lose history.  Report the
    # streamed file's row count instead.
    csv_path = out_dir / "results.csv"
    if csv_path.exists():
        try:
            import pandas as pd
            n_rows = len(pd.read_csv(csv_path, usecols=["case_id"]))
        except Exception:
            n_rows = len(result.df)
    else:
        result.df.to_csv(csv_path, index=False)
        n_rows = len(result.df)
    print(f"Streamed CSV: {csv_path} ({n_rows} rows)")
    if args.pool == "nature" and n_total >= LARGE_RUN_CELL_THRESHOLD:
        print("\nNext steps for a standalone formal Stage 2 run:")
        print(
            "  python scripts/run_pipeline.py --profile full --only sweep_audit "
            f"--sweep-dir {out_dir} --confirm-full-run"
        )
        print(
            "  python scripts/run_pipeline.py --profile full --from-stage 3 "
            f"--sweep-dir {out_dir} --confirm-full-run"
        )
    else:
        print(f"\nNext step for an ad-hoc/smoke run: python benchmark/scripts/generate_report.py {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
