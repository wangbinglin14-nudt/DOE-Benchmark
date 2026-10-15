"""
Helper script: physics-plausibility scan for every case.

Not part of the formal pytest suite; for audit/review use only.
"""

import warnings

import numpy as np
from scipy.stats.qmc import LatinHypercube

import doe_testbed as db

warnings.filterwarnings("ignore")


def audit_case(case, n: int = 100) -> dict:
    sampler = LatinHypercube(d=case.dim, seed=42)
    X = sampler.random(n)
    Y = case.evaluate(X)

    report = {
        "case_id": case.meta.case_id,
        "n": n,
        "dim_in": case.dim,
        "dim_out": case.n_outputs,
        "n_nan": int(np.sum(np.isnan(Y))),
        "n_inf": int(np.sum(np.isinf(Y))),
        "per_output": [],
    }

    for j, ov in enumerate(case.meta.output_variables):
        col = Y[:, j]
        col_ok = col[np.isfinite(col)]
        if len(col_ok) == 0:
            stats = {"min": None, "max": None, "mean": None}
        else:
            stats = {
                "min": float(col_ok.min()),
                "max": float(col_ok.max()),
                "mean": float(col_ok.mean()),
            }

        out_of_range = None
        if ov.expected_range is not None and len(col_ok) > 0:
            lo, hi = ov.expected_range
            out_of_range = int(np.sum((col_ok < lo) | (col_ok > hi)))

        report["per_output"].append(
            {
                "name": ov.name,
                "unit": ov.unit,
                "expected_range": ov.expected_range,
                **stats,
                "out_of_range": out_of_range,
                "failure_threshold": ov.failure_threshold,
                "failure_direction": ov.failure_direction,
            }
        )

    return report


def main():
    cases = db.list_cases()
    print(f"{'='*80}")
    print(f"  PHYSICS SANITY AUDIT — {len(cases)} cases, N=100 LHS each")
    print(f"{'='*80}\n")

    red_flags = []
    for m in cases:
        case = db.get_case(m.case_id)
        r = audit_case(case)
        flags = []
        if r["n_nan"] > 0:
            flags.append(f"NaN={r['n_nan']}")
        if r["n_inf"] > 0:
            flags.append(f"Inf={r['n_inf']}")
        for po in r["per_output"]:
            if po["out_of_range"] and po["out_of_range"] > int(0.05 * r["n"]):
                flags.append(
                    f"{po['name']}: {po['out_of_range']}/{r['n']} outside"
                    f" {po['expected_range']}  (actual=[{po['min']:.3g}, {po['max']:.3g}])"
                )

        marker = "[FAIL]" if flags else "[ OK ]"
        print(f"  {marker}  {m.case_id:<32s}  {m.dim_input:>2}D -> {m.dim_output}")
        for po in r["per_output"]:
            if po["min"] is not None:
                info = f"[{po['min']:>10.3g}, {po['max']:>10.3g}]"
                if po["expected_range"] is not None:
                    info += f"  expected={po['expected_range']}"
                print(f"       {po['name']:<16} {po['unit']:<8} {info}")
        if flags:
            red_flags.append((m.case_id, flags))
            for f in flags:
                print(f"       !! {f}")
        print()

    print("=" * 80)
    if red_flags:
        print(f"{len(red_flags)} case(s) with red flags:")
        for cid, flags in red_flags:
            print(f"  - {cid}: {flags}")
    else:
        print("All cases passed physical-range sanity check.")


if __name__ == "__main__":
    main()
