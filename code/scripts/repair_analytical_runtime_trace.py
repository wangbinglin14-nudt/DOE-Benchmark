"""Repair endpoint-consistent RSS metadata in the standalone D1 input.

The analytical phase predates the stricter runtime-row contract used by the
headline sweep. A small number of rows recorded ``rss_gb_peak`` just below
the start or end RSS because the monitor did not sample both endpoints. The
scientific D1 metrics do not use RSS, but the reusable-results reader does
validate the runtime trace. This script changes only those peaks to
``max(start, end, peak)`` and emits an audit JSON for every changed cell.

It is output-only by default. Use ``--in-place`` only when the original input
has been archived separately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "benchmark" / "results" / "ranking_reversal" / "analytical_results.csv"


def repair(input_path: Path, output_path: Path, audit_path: Path) -> dict[str, object]:
    frame = pd.read_csv(input_path, low_memory=False)
    required = {
        "case_id", "method", "budget_mult", "repeat",
        "rss_gb_start", "rss_gb_end", "rss_gb_peak",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"input is missing required columns: {missing}")

    rss = frame[["rss_gb_start", "rss_gb_end", "rss_gb_peak"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(rss.to_numpy()).all() or (rss < 0).to_numpy().any():
        raise ValueError("RSS columns must contain finite non-negative values")

    endpoint_max = rss[["rss_gb_start", "rss_gb_end"]].max(axis=1)
    changed = rss["rss_gb_peak"] < endpoint_max
    changes: list[dict[str, object]] = []
    for idx in frame.index[changed]:
        changes.append({
            "row_index": int(idx),
            "case_id": str(frame.at[idx, "case_id"]),
            "method": str(frame.at[idx, "method"]),
            "budget_mult": int(frame.at[idx, "budget_mult"]),
            "repeat": int(frame.at[idx, "repeat"]),
            "rss_gb_start": float(rss.at[idx, "rss_gb_start"]),
            "rss_gb_end": float(rss.at[idx, "rss_gb_end"]),
            "rss_gb_peak_before": float(rss.at[idx, "rss_gb_peak"]),
            "rss_gb_peak_after": float(endpoint_max.at[idx]),
        })

    repaired = frame.copy()
    repaired.loc[changed, "rss_gb_peak"] = endpoint_max.loc[changed]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    repaired.to_csv(output_path, index=False)

    audit = {
        "script": "scripts/repair_analytical_runtime_trace.py",
        "input": str(input_path),
        "output": str(output_path),
        "n_rows": int(len(frame)),
        "n_changed": int(changed.sum()),
        "scientific_columns_unchanged": True,
        "rule": "rss_gb_peak_after = max(rss_gb_start, rss_gb_end, rss_gb_peak_before)",
        "changes": changes,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?", default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--audit-output", type=Path, default=None)
    parser.add_argument("--in-place", action="store_true", help="replace input after archiving it externally")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.input if args.in_place else (
        args.output or args.input.with_name(args.input.stem + "_repaired.csv")
    )
    audit = args.audit_output or output.with_suffix(".repair_audit.json")
    result = repair(args.input, output, audit)
    print(f"[repair] rows={result['n_rows']} changed={result['n_changed']}")
    print(f"[repair] output={output}")
    print(f"[repair] audit={audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
