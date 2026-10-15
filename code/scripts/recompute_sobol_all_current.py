"""Run the uniform current-source Sobol recomputation for every registered case.

Results are written incrementally so a long-running sweep remains auditable if
an individual case fails.  This script only creates a new artifact; it never
edits the formal benchmark CSV or manuscript-derived files.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

from recompute_sobol_parallel import recompute


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmark/data/sobol_corrected_full_current_source.json",
    )
    args = parser.parse_args()

    import doe_testbed as db

    case_ids = sorted(meta.case_id for meta in db.list_cases())
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "sobol_corrected_current_source_v1",
        "created_by": "scripts/recompute_sobol_all_current.py",
        "source_commit": "1b691bf5c8884c1ce576d431c6059bdf84dd17ad",
        "salib_version": importlib.metadata.version("SALib"),
        "sampler": "SALib.sample.sobol.sample",
        "analyzer": "SALib.analyze.sobol.analyze",
        "calc_second_order": False,
        "N": args.n,
        "seed": args.seed,
        "workers": args.workers,
        "chunk_size": args.chunk_size,
        "case_count_expected": len(case_ids),
        "cases": [],
    }
    t0 = time.perf_counter()
    for index, case_id in enumerate(case_ids, start=1):
        print(f"[{index}/{len(case_ids)}] {case_id}", flush=True)
        row = recompute(case_id, args.n, args.seed, args.workers, args.chunk_size)
        payload["cases"] = [r for r in payload["cases"] if r["case_id"] != case_id]
        payload["cases"].append(row)
        payload["cases"].sort(key=lambda r: r["case_id"])
        payload["elapsed_seconds"] = time.perf_counter() - t0
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"    {row['seconds']:.1f}s", flush=True)
    payload["case_count"] = len(payload["cases"])
    payload["completed_seconds"] = time.perf_counter() - t0
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "case_count": payload["case_count"], "seconds": payload["completed_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
