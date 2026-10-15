#!/usr/bin/env python3
"""Merge a seven-case repair run into the Stage-2 startup snapshot.

The headline CSV is composed of the historical 27-case run plus a fresh
seven-case run.  This helper keeps the single-run-config contract honest by
requiring the protocol, method fingerprints, environment and runtime controls
to match, then replacing only the repaired case entries and recording the two
input config hashes in an explicit ``provenance_repair`` block.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPAIR_CASES = {
    "aircraft_conceptual_8d",
    "groundwater_contamination_6d",
    "rotor_bemt_6d",
    "solar_pv_yield_6d",
    "supersonic_intake_5d",
    "waverider_4d",
    "wind_turbine_power_6d",
}
EXPECTED_TOTAL_CELLS = 81_600
EXPECTED_REPAIR_CELLS = 16_800
EXPECTED_CANONICAL_CASES = 34
UNMERGED_KEYS = {"cases", "total_cells", "created_utc", "output_dir", "skip_existing"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"cannot parse {path}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    return value


def _case_map(config: dict, label: str) -> dict[str, dict]:
    cases = config.get("cases")
    if not isinstance(cases, list) or not all(isinstance(item, dict) for item in cases):
        raise SystemExit(f"{label} cases are missing or malformed")
    result = {str(item.get("case_id", "")): item for item in cases}
    if len(result) != len(cases) or "" in result:
        raise SystemExit(f"{label} cases contain duplicate or empty case_id values")
    return result


def merge(canonical_path: Path, repair_path: Path, output_path: Path) -> None:
    canonical = _load(canonical_path)
    repair = _load(repair_path)
    if canonical.get("artifact") != "doe-benchmark-stage2-run-config":
        raise SystemExit("canonical config has the wrong artifact marker")
    if repair.get("artifact") != "doe-benchmark-stage2-run-config":
        raise SystemExit("repair config has the wrong artifact marker")
    if int(canonical.get("total_cells", -1)) != EXPECTED_TOTAL_CELLS:
        raise SystemExit("canonical config is not the 81,600-cell archive config")
    if int(repair.get("total_cells", -1)) != EXPECTED_REPAIR_CELLS:
        raise SystemExit("repair config is not the expected 16,800-cell seven-case run")

    canonical_cases = _case_map(canonical, "canonical")
    repair_cases = _case_map(repair, "repair")
    if set(repair_cases) != REPAIR_CASES:
        raise SystemExit(
            f"repair config cases mismatch: expected {sorted(REPAIR_CASES)}, observed {sorted(repair_cases)}"
        )
    if not REPAIR_CASES.issubset(canonical_cases):
        raise SystemExit("canonical config does not contain every repair case")
    if len(canonical_cases) != EXPECTED_CANONICAL_CASES:
        raise SystemExit(
            f"canonical config must contain {EXPECTED_CANONICAL_CASES} cases, observed {len(canonical_cases)}"
        )

    for key in sorted((set(canonical) | set(repair)) - UNMERGED_KEYS):
        if key in {"provenance_repair"}:
            continue
        if canonical.get(key) != repair.get(key):
            raise SystemExit(f"canonical/repair config mismatch at key {key!r}; rerun with identical controls")

    merged = dict(canonical)
    merged["cases"] = [
        repair_cases[case_id] if case_id in REPAIR_CASES else canonical_cases[case_id]
        for case_id in canonical_cases
    ]
    merged["cases"] = sorted(merged["cases"], key=lambda item: str(item["case_id"]))
    merged["total_cells"] = EXPECTED_TOTAL_CELLS
    merged["created_utc"] = datetime.now(timezone.utc).isoformat()
    merged["skip_existing"] = False
    merged["provenance_repair"] = {
        "schema_version": 1,
        "script": "scripts/merge_run_config_repair.py",
        "merged_at_utc": merged["created_utc"],
        "historical_run_config_sha256": _sha256(canonical_path),
        "repair_run_config_sha256": _sha256(repair_path),
        "repair_cases": sorted(REPAIR_CASES),
        "historical_case_count": 27,
        "repair_case_count": 7,
    }

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=output_path.parent, suffix=".json") as fh:
        tmp = Path(fh.name)
        json.dump(merged, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    try:
        os.replace(tmp, output_path)
    finally:
        if tmp.exists():
            tmp.unlink()
    print(f"wrote {output_path}")
    print(f"sha256={_sha256(output_path)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=ROOT / "benchmark/results/full_benchmark/run_config.json")
    parser.add_argument("--repair", type=Path, required=True, help="run_config.json from the seven-case repair run")
    parser.add_argument("--output", type=Path, required=True, help="merged Stage-2 run_config.json")
    args = parser.parse_args()
    merge(args.canonical, args.repair, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
