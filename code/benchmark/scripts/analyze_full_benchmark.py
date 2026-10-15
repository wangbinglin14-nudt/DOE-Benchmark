"""
benchmark.scripts.analyze_full_benchmark
==========================================
Master post-completion analysis script.  Runs (in order):

  1. generate_report on the full benchmark   → REPORT.md + figs/
  2. compare_runs (pilot vs full)            → diff stats
  3. (optional) run_sensitivity_recovery     → sobol-recovery scores
  4. generate_report again to embed sensitivity findings

Pass --no-sensitivity to skip the (slow) sensitivity-recovery step.
Pass ``--full-dir <path>`` to point at a non-default sweep output
(e.g. ``benchmark/results/full_benchmark``).

Usage
-----
    python benchmark/scripts/analyze_full_benchmark.py
    python benchmark/scripts/analyze_full_benchmark.py --no-sensitivity
    python benchmark/scripts/analyze_full_benchmark.py --skip-pilot-compare
    python benchmark/scripts/analyze_full_benchmark.py \\
        --full-dir benchmark/results/full_benchmark
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
import time
import warnings
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

DEFAULT_FULL_DIR = ROOT / "benchmark" / "results" / "full_benchmark"
DEFAULT_PILOT_DIR = ROOT / "benchmark" / "results" / "extended_pilot"
DEFAULT_SENS_DIR = ROOT / "benchmark" / "results" / "sensitivity_recovery"


def _run_script(script: str, *args: str) -> int:
    """Run a benchmark script as a subprocess and stream output line-by-line.

    Uses ``python -u`` to avoid stdout buffering when piped through
    Tee-Object on Windows.
    """
    cmd = [sys.executable, "-u", "-W", "ignore",
           str(ROOT / "benchmark" / "scripts" / script), *args]
    print("\n" + "=" * 72)
    print(f"$ {' '.join(cmd[3:])}")
    print("=" * 72, flush=True)
    t0 = time.perf_counter()
    env = {**__import__("os").environ,
           "PYTHONIOENCODING": "utf-8",
           "PYTHONUNBUFFERED": "1"}
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
    dt = time.perf_counter() - t0
    print(f"[{script} finished in {dt/60:.1f} min, rc={proc.returncode}]",
          flush=True)
    return proc.returncode


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="analyze_full_benchmark",
        description=(
            "Master post-completion analysis: generate_report + "
            "compare_runs + sensitivity_recovery."
        ),
    )
    p.add_argument(
        "--full-dir",
        type=Path,
        default=DEFAULT_FULL_DIR,
        help=(
            "Full-sweep results directory containing results.csv "
            f"(default: {DEFAULT_FULL_DIR.relative_to(ROOT)})."
        ),
    )
    p.add_argument(
        "--pilot-dir",
        type=Path,
        default=DEFAULT_PILOT_DIR,
        help=(
            "Pilot results directory for compare_runs "
            f"(default: {DEFAULT_PILOT_DIR.relative_to(ROOT)})."
        ),
    )
    p.add_argument(
        "--sens-dir",
        type=Path,
        default=None,
        help=(
            "Sensitivity-recovery output directory.  If omitted, "
            "defaults to ``<full-dir>/sensitivity_recovery`` so each "
            "sweep gets its own colocated artefacts; pre-2026-05 "
            "callers depending on the legacy global directory "
            f"({DEFAULT_SENS_DIR.relative_to(ROOT)}) should pass "
            "``--sens-dir`` explicitly."
        ),
    )
    p.add_argument(
        "--no-sensitivity",
        action="store_true",
        help="Skip the (slow) sensitivity-recovery step.",
    )
    p.add_argument(
        "--skip-pilot-compare",
        action="store_true",
        help="Skip the pilot-vs-full comparison even if a pilot exists.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None):
    args = _parse_args(argv)
    full_dir = args.full_dir.resolve()
    pilot_dir = args.pilot_dir.resolve()
    sens_dir = (
        args.sens_dir.resolve()
        if args.sens_dir is not None
        else (full_dir / "sensitivity_recovery").resolve()
    )
    do_sensitivity = not args.no_sensitivity
    do_pilot_compare = not args.skip_pilot_compare

    if not (full_dir / "results.csv").exists():
        print(f"[error] {full_dir/'results.csv'} not found.  "
              "Run benchmark/scripts/run_full_benchmark.py first.")
        sys.exit(1)

    print(f"\nFull benchmark CSV: {full_dir/'results.csv'}")
    import pandas as pd
    df = pd.read_csv(full_dir / "results.csv")
    print(f"  {len(df)} rows, "
          f"{df['case_id'].nunique()} cases, "
          f"{df['method'].nunique()} methods, "
          f"{df['budget_mult'].nunique()} budgets, "
          f"{df['repeat'].nunique()} repeats")

    # ── Step 1: rigorous report ───────────────────────────────────────
    rc = _run_script("generate_report.py", str(full_dir))
    if rc != 0:
        print("[error] generate_report failed; aborting downstream steps.")
        sys.exit(rc)

    # ── Step 2: pilot vs full comparison ──────────────────────────────
    if do_pilot_compare and (pilot_dir / "results.csv").exists():
        _run_script("compare_runs.py", str(pilot_dir), str(full_dir))
    else:
        print("\n[skip] pilot-vs-full comparison "
              "(no pilot or --skip-pilot-compare)")

    # ── Step 3: sensitivity recovery ──────────────────────────────────
    if do_sensitivity:
        if not (sens_dir / "sensitivity_recovery.csv").exists():
            _run_script(
                "run_sensitivity_recovery.py",
                "--output-dir", str(sens_dir),
            )
        else:
            print(f"\n[skip] sensitivity recovery already at "
                  f"{sens_dir/'sensitivity_recovery.csv'}")

        # Re-run report to embed sensitivity findings
        _run_script("generate_report.py", str(full_dir))
    else:
        print("\n[skip] sensitivity recovery (--no-sensitivity)")

    print("\n" + "=" * 72)
    print(f"All analyses complete.  Headline: {full_dir/'report/REPORT.md'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
