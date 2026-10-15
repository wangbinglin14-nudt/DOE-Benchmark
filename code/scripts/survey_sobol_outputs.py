"""Identify which Sobol-reference output index each case uses.

Some cases recorded their Sobol indices on a non-default output (i.e.
``output_names[k]`` for ``k > 0``). The convention is to embed the
output name in ``meta.sobol_computation`` as the ``..._<name>_output``
suffix; everything else defaults to ``output_names[0]``.
"""
from __future__ import annotations

import doe_testbed as db


def detect_output_index(meta) -> int:
    """Return the output index whose Sobol indices were stored, defaulting to 0."""
    s = meta.sobol_computation or ""
    if "_output" not in s:
        return 0
    # Strip the trailing "_output" then find the last output name that suffixes it
    head = s[: -len("_output")]
    for k, name in enumerate(meta.output_names):
        if head.endswith("_" + name) or head.endswith(name):
            return k
    return 0


def main() -> None:
    print("Sobol reference output assignment per case:")
    print()
    for cid in db.list_cases(return_ids=True):
        meta = db.get_case(cid).meta
        if not meta.sobol_S1:
            continue
        idx = detect_output_index(meta)
        if idx != 0:
            print(
                f"  {cid:32s} output[{idx}]='{meta.output_names[idx]}'  "
                f"(outputs={meta.output_names})  "
                f"sobol_computation={meta.sobol_computation!r}"
            )


if __name__ == "__main__":
    main()
