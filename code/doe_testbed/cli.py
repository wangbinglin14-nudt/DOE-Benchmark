"""
doe_testbed.cli
==================
Command-line entry point (available as ``doe-testbed`` after pip install).
"""

from __future__ import annotations

import argparse
import sys
import warnings

warnings.filterwarnings("ignore")

# Windows GBK terminal compatibility: force UTF-8 output
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        import io

        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )
    except Exception:
        pass


def cmd_list(args):
    """List all registered cases."""
    from doe_testbed import list_cases

    cases = list_cases()

    from collections import defaultdict

    by_domain = defaultdict(list)
    for m in cases:
        by_domain[m.domain].append(m)

    print(f"\nDOE Testbed  --  {len(cases)} cases\n")
    for domain, metas in sorted(by_domain.items()):
        print(f"  [{domain}]")
        for m in sorted(metas, key=lambda x: x.case_id):
            scenario_str = ",".join(m.applicable_scenarios)
            print(f"    {m.case_id:<35}  {m.dim_input:>2}D->{m.dim_output}  [{scenario_str}]")
    print()


def cmd_info(args):
    """Print detailed information for a given case."""
    from doe_testbed import get_case

    case = get_case(args.case_id)
    m = case.meta
    print(f"\n{'='*60}")
    print(f"  {m.case_id}")
    print(f"{'='*60}")
    print(f"  Name         : {m.name}")
    print(f"  Domain       : {m.domain}")
    print(f"  Dimensions   : {m.dim_input}D -> {m.dim_output}")
    print(f"  Scenarios    : {m.applicable_scenarios}")
    print(f"  Governing Eq : {m.governing_equation[:80]}")
    print("\n  Input variables:")
    for v in m.input_variables:
        print(f"    {v.symbol:>8}  {v.name:<22}  [{v.lower:.3g}, {v.upper:.3g}] {v.unit}")
    print("\n  Output variables:")
    for v in m.output_variables:
        print(f"    {v.symbol:>8}  {v.name:<22}  unit={v.unit}")
    print(f"\n  Bounds lower : {m.bounds_lower}")
    print(f"  Bounds upper : {m.bounds_upper}")
    print()


def cmd_check(args):
    """Run check_implementation self-tests."""
    from doe_testbed import get_case, list_cases

    if args.case_id == "all":
        cases_ids = [m.case_id for m in list_cases()]
    else:
        cases_ids = [args.case_id]

    passed = failed = 0
    for cid in cases_ids:
        case = get_case(cid)
        result = case.check_implementation(verbose=False)
        bool_items = {k: v for k, v in result.items() if isinstance(v, bool)}
        fail_items = [k for k, v in bool_items.items() if not v]
        if len(fail_items) == 0:
            print(f"  PASS  {cid}")
            passed += 1
        else:
            print(f"  FAIL  {cid}  ({fail_items})")
            failed += 1

    print(f"\n{passed} passed, {failed} failed  (total {passed+failed})")


def cmd_coverage(args):
    """Print VS-system coverage matrix and distribution statistics."""
    from doe_testbed import list_cases
    from doe_testbed.base.scenarios import VS_ALL

    metas = list_cases()
    metas_sorted = sorted(metas, key=lambda m: (m.domain, m.case_id))

    # ── VS coverage matrix ─────────────────────────────────────────────────
    print()
    print(f"VS-System Coverage Matrix  ({len(metas)} cases)")
    header = f"{'case_id':<32} {'dim':>4} {'cost':>5}  " + " ".join(f"{s[-1]:>2}" for s in VS_ALL)
    print(header)
    print("-" * len(header))

    col_counts = {s: 0 for s in VS_ALL}
    for m in metas_sorted:
        marks = []
        for s in VS_ALL:
            if s in m.applicable_scenarios:
                marks.append(" #")
                col_counts[s] += 1
            else:
                marks.append("  ")
        print(f"{m.case_id:<32} {m.dim_input:>4} {m.cost_level:>5}  " + " ".join(marks))

    print("-" * len(header))
    totals = f"{'coverage (count)':<32} {'':>4} {'':>5}  " + " ".join(
        f"{col_counts[s]:>2}" for s in VS_ALL
    )
    print(totals)
    print()

    # ── Dimension distribution ────────────────────────────────────────────────
    from collections import Counter

    dim_buckets = Counter()
    for m in metas:
        if m.dim_input <= 4:
            dim_buckets["≤4D"] += 1
        elif m.dim_input <= 6:
            dim_buckets["5–6D"] += 1
        elif m.dim_input <= 9:
            dim_buckets["7–9D"] += 1
        else:
            dim_buckets["≥10D"] += 1
    print("Dimension buckets:")
    for k in ["≤4D", "5–6D", "7–9D", "≥10D"]:
        print(f"  {k:<6} : {dim_buckets[k]:>2}")

    # ── Cost distribution ───────────────────────────────────────────────────
    cost_cnt = Counter(m.cost_level for m in metas)
    print("\nCost buckets:")
    for k in ["L0", "L1", "L2", "L3"]:
        print(f"  {k:<3} : {cost_cnt.get(k, 0):>2}")

    # ── Domain distribution ─────────────────────────────────────────────────
    dom_cnt = Counter(m.domain for m in metas)
    print("\nDomain buckets:")
    for k, v in sorted(dom_cnt.items()):
        print(f"  {k:<20} : {v:>2}")

    # ── Coverage completeness check ─────────────────────────────────────────
    min_needed = 3
    low = [s for s, c in col_counts.items() if c < min_needed]
    if low:
        print(f"\n[WARN] Scenarios with < {min_needed} representative cases: {low}")
    else:
        print(f"\n[OK] Every VS scenario has at least {min_needed} representative cases.")


def main():
    parser = argparse.ArgumentParser(
        prog="doe-testbed",
        description="DOE Testbed CLI — engineering test problem library",
    )
    sub = parser.add_subparsers(dest="command")

    # list
    p_list = sub.add_parser("list", help="List all registered test problems")
    p_list.set_defaults(func=cmd_list)

    # info
    p_info = sub.add_parser("info", help="Show detailed info for a case")
    p_info.add_argument("case_id")
    p_info.set_defaults(func=cmd_info)

    # check
    p_check = sub.add_parser("check", help="Run check_implementation")
    p_check.add_argument("case_id", default="all", nargs="?", help="case_id or 'all' (default)")
    p_check.set_defaults(func=cmd_check)

    # coverage
    p_cov = sub.add_parser("coverage", help="Print VS-System coverage matrix + stats")
    p_cov.set_defaults(func=cmd_coverage)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
