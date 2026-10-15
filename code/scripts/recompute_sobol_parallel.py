"""Recompute current-source Sobol metadata with process-parallel evaluation.

This utility is intentionally separate from the historical cross-validation
script.  It records the exact SALib configuration, case source hash, output
name, and wall time so a corrected artifact can be audited before it is used
to regenerate the manuscript.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np


def _evaluate_chunk(args: tuple[str, np.ndarray]) -> np.ndarray:
    case_id, x = args
    import doe_testbed as db

    return np.asarray(db.get_case(case_id).evaluate(x), dtype=float)


def _source_hash(case_id: str) -> tuple[str, str]:
    import doe_testbed as db

    case = db.get_case(case_id)
    path = Path(case.__class__.__module__.replace(".", "/") + ".py")
    path = Path(__file__).resolve().parents[1] / path
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return str(path.relative_to(Path(__file__).resolve().parents[1])), digest


def recompute(case_id: str, n: int, seed: int, workers: int, chunk_size: int) -> dict[str, Any]:
    from SALib.analyze import sobol as sobol_analyze
    from SALib.sample import sobol as sobol_sample
    import doe_testbed as db

    case = db.get_case(case_id)
    meta = case.meta
    problem = {
        "num_vars": meta.dim_input,
        "names": meta.input_names,
        "bounds": [[0.0, 1.0]] * meta.dim_input,
    }
    x = sobol_sample.sample(problem, n, calc_second_order=False, seed=seed)
    chunks = [x[i : i + chunk_size] for i in range(0, len(x), chunk_size)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        ys = list(pool.map(_evaluate_chunk, [(case_id, c) for c in chunks]))
    y = np.concatenate(ys, axis=0)
    out_idx = 0
    s = meta.sobol_computation or ""
    if "_output" in s:
        head = s[: -len("_output")]
        for i, name in enumerate(meta.output_names):
            if head.endswith("_" + name) or head.endswith(name):
                out_idx = i
                break
    si = sobol_analyze.analyze(
        problem,
        y[:, out_idx],
        calc_second_order=False,
        seed=seed,
        print_to_console=False,
    )
    rel, digest = _source_hash(case_id)
    return {
        "case_id": case_id,
        "N": n,
        "seed": seed,
        "workers": workers,
        "chunk_size": chunk_size,
        "n_evals": int(len(x)),
        "output": meta.output_names[out_idx],
        "output_index": out_idx,
        "source_file": rel,
        "source_sha256": digest,
        "seconds": time.perf_counter() - t0,
        "S1": {k: float(v) for k, v in zip(meta.input_names, si["S1"], strict=False)},
        "ST": {k: float(v) for k, v in zip(meta.input_names, si["ST"], strict=False)},
        "S1_conf": {k: float(v) for k, v in zip(meta.input_names, si["S1_conf"], strict=False)},
        "ST_conf": {k: float(v) for k, v in zip(meta.input_names, si["ST_conf"], strict=False)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_id")
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    row = recompute(args.case_id, args.n, args.seed, args.workers, args.chunk_size)
    payload = {args.case_id: row}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
