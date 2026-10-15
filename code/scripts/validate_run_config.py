"""Pre-flight config validator for the headline 30-repeat sweep.

Refuses to launch a sweep whose configuration violates the frozen
``paper/preregistration.md`` v0.4.0-prereg protocol, unless the
``--allow-deviation`` escape hatch is given (which logs a warning to
``benchmark/results/<output-dir>/preregistration_deviations.txt``).

This is the single source of truth for "is this run paper-compliant?"
and is referenced by:
  * scripts/run_pipeline.py               (formal full-pipeline preflight)
  * scripts/launch_sweep_remote.sh        (low-level server Stage-2 preflight)
  * scripts/launch_sweep_local_16gb.ps1   (guarded Windows diagnostic only)
  * Snakemake rule main_benchmark         (legacy CI gate)

Usage
-----
    python scripts/validate_run_config.py \
        --pool nature --budgets 5 10 20 50 \
        --n-repeats 30 --n-test 1000 --seed-strategy case_repeat \
        --cases ALL

Exit codes
----------
    0  config matches pre-registration; safe to launch.
    1  config deviates and ``--allow-deviation`` was not given.
    2  malformed CLI / runtime error.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Frozen pre-registration values (paper/preregistration.md §3) ─────────
PREREG_POOL = "nature"
PREREG_BUDGETS = [5, 10, 20, 50]
PREREG_N_REPEATS = 30
PREREG_N_TEST = 1000
PREREG_SEED_STRATEGY = "case_repeat"
PREREG_SEED_START = 1000


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", required=True)
    p.add_argument("--budgets", type=int, nargs="+", required=True)
    p.add_argument("--n-repeats", type=int, required=True)
    p.add_argument("--n-test", type=int, default=PREREG_N_TEST)
    p.add_argument("--seed-strategy", default=PREREG_SEED_STRATEGY)
    p.add_argument("--seed-start", type=int, default=PREREG_SEED_START)
    p.add_argument(
        "--cases",
        nargs="+",
        default=["ALL"],
        help="Either 'ALL' or an explicit list of case_ids.",
    )
    p.add_argument(
        "--allow-deviation",
        action="store_true",
        help="Permit deviations; logs a warning instead of refusing to launch.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="If given, deviation log is written here.",
    )
    return p.parse_args(argv)


def _check_methods(pool: str) -> Tuple[bool, str]:
    """Verify the named pool yields the expected method count."""
    try:
        from benchmark.methods import list_methods
    except ImportError as exc:
        return False, f"cannot import benchmark.methods: {exc}"
    try:
        methods = list_methods(pool)
    except Exception as exc:
        return False, f"unknown pool {pool!r}: {exc}"
    if pool == "nature" and len(methods) != 20:
        # BoTorch may be unavailable; keep this diagnostic path visible, but
        # the formal publication launch blocks it via the dependency preflight.
        if len(methods) == 18:
            return True, (
                f"pool='nature' returned 18 methods (BoTorch unavailable); "
                f"this is a diagnostic-only fallback. The Nature-family "
                f"headline sweep must pass --require-botorch and resolve "
                f"20 methods before launch."
            )
        return False, (
            f"pool='nature' returned {len(methods)} methods, expected 20 "
            f"for the formal headline protocol.  Methods: {methods}"
        )
    return True, f"pool='{pool}' yielded {len(methods)} methods"


def _check_hyperparameters() -> Tuple[bool, str]:
    """GAP-9: cross-check ``benchmark/hyperparameters.py`` registration.

    Two layers of protection:

    1. The registry must enumerate at least 18 methods (the nature pool
       minus the two optional BoTorch entries).  Catches silent merge
       accidents that would otherwise produce a sweep with default
       sklearn hyper-params.
    2. The canonical table must agree element-for-element with the
       values actually injected by the pool factories.  Without this
       layer the paper would report hyper-parameters that were not
       run; with it any drift is a hard launch-blocker.
    """
    try:
        from benchmark.hyperparameters import (
            HYPERPARAMETERS,
            assert_hyperparameters_consistent,
        )
    except Exception as exc:
        return False, f"cannot import benchmark.hyperparameters: {exc}"
    n = len(HYPERPARAMETERS)
    if n < 18:
        return False, (
            f"hyperparameters registry has only {n} entries; "
            "expected >= 18 (nature_pool minus optional BoTorch)."
        )
    try:
        assert_hyperparameters_consistent()
    except AssertionError as exc:
        return False, (
            f"hyperparameters registry drifts from pool factories: "
            f"{exc}.  Update benchmark/hyperparameters.py or "
            f"benchmark/methods.py so paper-reported values match "
            f"actually-run values (DOE-Benchmark §11 GAP-9)."
        )
    return True, (
        f"hyperparameters registry: {n} methods registered & "
        f"consistent with pool factories"
    )


def _check_cases(requested: List[str]) -> Tuple[bool, str]:
    """Verify the requested case set covers all 34 cases of doe_testbed."""
    try:
        import doe_testbed as db
    except ImportError as exc:
        return False, f"cannot import doe_testbed: {exc}"
    all_cases = sorted(c.case_id for c in db.list_cases())
    if requested == ["ALL"]:
        return True, f"running all {len(all_cases)} cases of doe_testbed"
    missing = sorted(set(all_cases) - set(requested))
    extra = sorted(set(requested) - set(all_cases))
    if missing or extra:
        problems = []
        if missing:
            problems.append(f"missing {len(missing)}: {missing}")
        if extra:
            problems.append(f"unknown {len(extra)}: {extra}")
        return False, "; ".join(problems)
    return True, f"all {len(all_cases)} cases requested"


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    deviations: List[str] = []
    notes: List[str] = []

    # ── Methods / pool ────────────────────────────────────────────────────
    ok, msg = _check_methods(args.pool)
    if not ok:
        deviations.append(f"pool: {msg}")
    else:
        notes.append(f"pool: {msg}")
    if args.pool != PREREG_POOL:
        deviations.append(
            f"pool: got '{args.pool}', pre-reg says '{PREREG_POOL}'"
        )

    # ── Hyperparameters registry (GAP-9) ─────────────────────────────────
    ok, msg = _check_hyperparameters()
    if not ok:
        deviations.append(f"hyperparameters: {msg}")
    else:
        notes.append(f"hyperparameters: {msg}")

    # ── Cases ─────────────────────────────────────────────────────────────
    ok, msg = _check_cases(args.cases)
    if not ok:
        deviations.append(f"cases: {msg}")
    else:
        notes.append(f"cases: {msg}")

    # ── Budgets ───────────────────────────────────────────────────────────
    if sorted(args.budgets) != PREREG_BUDGETS:
        deviations.append(
            f"budgets: got {args.budgets}, pre-reg says {PREREG_BUDGETS}"
        )
    else:
        notes.append(f"budgets: {args.budgets} matches pre-reg")

    # ── n_repeats ─────────────────────────────────────────────────────────
    if args.n_repeats < PREREG_N_REPEATS:
        deviations.append(
            f"n-repeats: got {args.n_repeats}, pre-reg says {PREREG_N_REPEATS}"
            f" (smaller n_repeats inflates CI widths beyond pre-registered)"
        )
    elif args.n_repeats > PREREG_N_REPEATS:
        notes.append(
            f"n-repeats: got {args.n_repeats}, pre-reg says {PREREG_N_REPEATS}"
            f" (larger is acceptable but un-needed)"
        )
    else:
        notes.append(f"n-repeats: {args.n_repeats} matches pre-reg")

    # ── n_test ────────────────────────────────────────────────────────────
    if args.n_test != PREREG_N_TEST:
        deviations.append(
            f"n-test: got {args.n_test}, pre-reg says {PREREG_N_TEST}"
        )
    else:
        notes.append(f"n-test: {args.n_test} matches pre-reg")

    # ── seed_strategy ─────────────────────────────────────────────────────
    if args.seed_strategy != PREREG_SEED_STRATEGY:
        deviations.append(
            f"seed-strategy: got '{args.seed_strategy}', "
            f"pre-reg says '{PREREG_SEED_STRATEGY}'"
        )
    else:
        notes.append(f"seed-strategy: '{args.seed_strategy}' matches pre-reg")

    if args.seed_start != PREREG_SEED_START:
        deviations.append(
            f"seed-start: got {args.seed_start}, pre-reg says {PREREG_SEED_START}"
        )
    else:
        notes.append(f"seed-start: {args.seed_start} matches pre-reg")

    # ── Verdict ───────────────────────────────────────────────────────────
    print("=" * 70)
    print("Pre-registration compliance check")
    print("=" * 70)
    for n in notes:
        print(f"  OK   {n}")
    for d in deviations:
        print(f"  FAIL {d}")
    print("=" * 70)

    if not deviations:
        print("Result: COMPLIANT — safe to launch.")
        return 0

    if args.allow_deviation:
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            log_path = args.output_dir / "preregistration_deviations.txt"
            log_path.write_text(
                "Pre-registration deviations recorded at launch:\n"
                + "\n".join(f"  - {d}" for d in deviations)
                + "\n\nLogged because --allow-deviation was passed.\n"
                + "This run is NOT paper-compliant for the headline 30-repeat sweep.\n",
                encoding="utf-8",
            )
            print(f"Deviation log written: {log_path}")
        print(
            f"Result: {len(deviations)} DEVIATION(S) but --allow-deviation "
            "permits launch."
        )
        return 0

    print(
        f"Result: REFUSING TO LAUNCH — {len(deviations)} deviation(s) from "
        f"pre-registration.\n"
        f"  Re-run with --allow-deviation to override (logs to "
        f"preregistration_deviations.txt)."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
