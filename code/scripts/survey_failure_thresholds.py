"""Survey every case for failure thresholds and print a Markdown table.

Used during documentation work to make sure every failure-boundary case
is annotated with provenance (engineering standard / regulation /
textbook clause) in docs/failure_thresholds.md.
"""
from __future__ import annotations

import doe_testbed as db


def main() -> None:
    cases = db.list_cases(has_failure_boundary=True, return_ids=True)
    print(f"Total cases with failure boundary: {len(cases)}\n")
    print("| case_id | output | threshold | direction | unit |")
    print("|---|---|---:|---|---|")
    for cid in cases:
        meta = db.get_case(cid).meta
        for v in meta.output_variables:
            if v.failure_threshold is None:
                continue
            print(
                f"| `{cid}` | `{v.name}` | {v.failure_threshold} "
                f"| {v.failure_direction} | {v.unit or '-'} |"
            )


if __name__ == "__main__":
    main()
