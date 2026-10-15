"""Assemble the current-source Sobol estimates into one auditable artifact.

The 29-case diagnostic predates the expensive-case reruns and stores its
estimates under ``S1_alt``/``ST_alt``.  The five expensive cases are emitted by
``recompute_sobol_parallel.py`` with full confidence intervals.  This utility
combines them without changing either source dataset and refuses to write an
artifact unless all 34 registered cases are present exactly once with the
same Saltelli N and seed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _source_hash(case_id: str) -> tuple[str, str]:
    import doe_testbed as db

    case = db.get_case(case_id)
    path = ROOT / (case.__class__.__module__.replace(".", "/") + ".py")
    return str(path.relative_to(ROOT)), hashlib.sha256(path.read_bytes()).hexdigest()


def _registered_cases() -> list[str]:
    import doe_testbed as db

    return sorted(meta.case_id for meta in db.list_cases())


def _load_expensive(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in sorted(path.glob("*.json")):
        payload = json.loads(item.read_text(encoding="utf-8"))
        for case_id, row in payload.items():
            # Keep only the uniform configuration; older exploratory candidates
            # in this directory are intentionally ignored.
            if int(row.get("N", -1)) != 1024 or int(row.get("seed", -1)) != 20260831:
                continue
            if case_id in out:
                raise ValueError(f"duplicate expensive case: {case_id}")
            out[case_id] = row
    return out


def _load_diagnostic(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for row in payload["rows"]:
        case_id = row["case_id"]
        if case_id in out:
            raise ValueError(f"duplicate diagnostic case: {case_id}")
        out[case_id] = row
    return out


def assemble(diagnostic: Path, expensive_dir: Path) -> dict[str, Any]:
    diag = _load_diagnostic(diagnostic)
    expensive = _load_expensive(expensive_dir)
    registered = _registered_cases()
    missing = sorted(set(registered) - set(diag) - set(expensive))
    extra = sorted((set(diag) | set(expensive)) - set(registered))
    if missing or extra:
        raise ValueError(f"case coverage mismatch: missing={missing}, extra={extra}")

    rows: list[dict[str, Any]] = []
    for case_id in registered:
        if case_id in diag and case_id in expensive:
            raise ValueError(f"case appears in both diagnostic and expensive sets: {case_id}")
        if case_id in expensive:
            src = dict(expensive[case_id])
            provenance = "parallel_recompute"
            n = int(src["N"])
            seed = int(src["seed"])
            if n != 1024 or seed != 20260831:
                raise ValueError(f"unexpected configuration for {case_id}: N={n}, seed={seed}")
        else:
            src = diag[case_id]
            provenance = "current_source_diagnostic"
            n = int(src["n_evals_alt"]) // (int(src["dim"]) + 2)
            seed = 20260831
            if n != 1024:
                raise ValueError(f"unexpected diagnostic N for {case_id}: {n}")
            src = {
                "case_id": case_id,
                "N": n,
                "seed": seed,
                "n_evals": int(src["n_evals_alt"]),
                "output": src["ref_output_name"],
                "output_index": int(src["ref_output_index"]),
                "S1": src["S1_alt"],
                "ST": src["ST_alt"],
                "S1_conf": None,
                "ST_conf": None,
                "diagnostic_median_CI_S1": src["median_CI_S1"],
                "diagnostic_median_CI_ST": src["median_CI_ST"],
                "diagnostic_wall_seconds": src["wall_seconds"],
            }
        source_file, source_sha256 = _source_hash(case_id)
        if src.get("source_sha256") and src["source_sha256"] != source_sha256:
            raise ValueError(f"source hash mismatch for {case_id}")
        src["source_file"] = source_file
        src["source_sha256"] = source_sha256
        src["provenance"] = provenance
        rows.append(src)

    return {
        "schema": "sobol_corrected_current_source_v1",
        "created_by": "scripts/assemble_sobol_corrected_artifact.py",
        "source_commit": "1b691bf5c8884c1ce576d431c6059bdf84dd17ad",
        "salib_version": importlib.metadata.version("SALib"),
        "sampler": "SALib.sample.sobol.sample",
        "analyzer": "SALib.analyze.sobol.analyze",
        "calc_second_order": False,
        "N": 1024,
        "seed": 20260831,
        "case_count": len(rows),
        "cases": rows,
        "confidence_interval_note": (
            "Full per-variable confidence intervals are present for parallel_recompute "
            "rows. Diagnostic rows retain median confidence widths only and require a "
            "full rerun before use for uncertainty claims."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostic",
        type=Path,
        default=ROOT / "_archive_20260909/sobol-diagnostics/benchmark/data/sobol_cross_validation_current_1024.json",
    )
    parser.add_argument(
        "--expensive-dir",
        type=Path,
        default=ROOT / "_archive_20260909/sobol-diagnostics/benchmark/data/sobol_corrected_candidate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "_archive_20260909/sobol-diagnostics/benchmark/data/sobol_corrected_current_source.json",
    )
    args = parser.parse_args()
    payload = assemble(args.diagnostic, args.expensive_dir)
    if payload["case_count"] != 34:
        raise ValueError(f"expected 34 cases, found {payload['case_count']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "case_count": payload["case_count"], "N": payload["N"], "seed": payload["seed"], "salib_version": payload["salib_version"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
