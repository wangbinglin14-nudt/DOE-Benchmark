"""Inspect every case's failure_condition string and per-output threshold metadata."""
from __future__ import annotations

import io
import sys

import doe_testbed as db

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def main() -> None:
    for cid in db.list_cases(has_failure_boundary=True, return_ids=True):
        meta = db.get_case(cid).meta
        print(f"### {cid}")
        print(f"  failure_condition: {meta.failure_condition}")
        for v in meta.output_variables:
            if v.failure_threshold is None:
                continue
            unit = v.unit or "-"
            desc = (v.description or "")[:100]
            print(
                f"  {v.name} = {v.failure_threshold} {unit} "
                f"({v.failure_direction})  // {desc}"
            )
        print()


if __name__ == "__main__":
    main()
