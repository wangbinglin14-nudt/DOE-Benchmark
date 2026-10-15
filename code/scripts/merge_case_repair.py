#!/usr/bin/env python3
"""Merge a provenance-repair case sweep into the canonical headline CSV.

The June headline archive contains complete rows for all 34 cases, but seven
case source fingerprints cannot be reproduced from the available snapshots.
This utility accepts a separately-run repair CSV, validates its exact cell
grid and current source fingerprints, removes the corresponding historical
rows, and writes a deterministic replacement archive.  It never overwrites
the input unless ``--output`` explicitly names the same path and
``--in-place`` is supplied.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import os
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPAIR_CASES = (
    "aircraft_conceptual_8d",
    "groundwater_contamination_6d",
    "rotor_bemt_6d",
    "solar_pv_yield_6d",
    "supersonic_intake_5d",
    "waverider_4d",
    "wind_turbine_power_6d",
)
EXPECTED_BUDGETS = (5, 10, 20, 50)
EXPECTED_REPEATS = 30
EXPECTED_METHODS = 20
KEYS = ["case_id", "method", "budget_mult", "repeat"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_fingerprints() -> dict[str, tuple[str, str]]:
    import doe_testbed as db

    fingerprints: dict[str, tuple[str, str]] = {}
    for case_id in REPAIR_CASES:
        case = db.get_case(case_id)
        module = importlib.import_module(type(case).__module__)
        source = Path(str(module.__file__)).resolve()
        if not source.is_file():
            raise SystemExit(f"missing source for {case_id}: {source}")
        try:
            relative = source.relative_to(ROOT).as_posix()
        except ValueError:
            relative = source.as_posix()
        fingerprints[case_id] = (relative, _sha256(source))
    return fingerprints


def _validate_repair(df: pd.DataFrame) -> None:
    missing = [key for key in KEYS if key not in df.columns]
    if missing:
        raise SystemExit(f"repair CSV missing key columns: {missing}")
    cases = tuple(sorted(df["case_id"].dropna().unique()))
    if cases != tuple(sorted(REPAIR_CASES)):
        raise SystemExit(f"repair CSV cases mismatch: expected {REPAIR_CASES}, observed {cases}")
    methods = df["method"].dropna().unique()
    if len(methods) != EXPECTED_METHODS:
        raise SystemExit(f"repair CSV must contain {EXPECTED_METHODS} methods, observed {len(methods)}")
    budgets = tuple(sorted(pd.to_numeric(df["budget_mult"], errors="raise").unique()))
    if budgets != EXPECTED_BUDGETS:
        raise SystemExit(f"repair CSV budgets mismatch: expected {EXPECTED_BUDGETS}, observed {budgets}")
    repeats = tuple(sorted(pd.to_numeric(df["repeat"], errors="raise").unique()))
    if repeats != tuple(range(EXPECTED_REPEATS)):
        raise SystemExit(f"repair CSV repeats mismatch: expected 0..{EXPECTED_REPEATS - 1}, observed {repeats}")
    expected_rows = len(REPAIR_CASES) * EXPECTED_METHODS * len(EXPECTED_BUDGETS) * EXPECTED_REPEATS
    if len(df) != expected_rows:
        raise SystemExit(f"repair CSV row count mismatch: expected {expected_rows}, observed {len(df)}")
    dup = df.duplicated(KEYS, keep=False)
    if bool(dup.any()):
        raise SystemExit(f"repair CSV contains {int(dup.sum())} duplicate key rows")
    counts = df.groupby(KEYS, dropna=False).size()
    if len(counts) != expected_rows or not bool((counts == 1).all()):
        raise SystemExit("repair CSV does not contain exactly one row per declared cell")
    missing_provenance = [
        name for name in ("case_source_path", "case_source_sha256") if name not in df.columns
    ]
    if missing_provenance:
        raise SystemExit(f"repair CSV lacks provenance columns: {missing_provenance}")
    expected_fingerprints = _source_fingerprints()
    observed = df.groupby("case_id")["case_source_sha256"].unique().to_dict()
    paths = df.groupby("case_id")["case_source_path"].unique().to_dict()
    for case_id, (expected_path, expected_hash) in expected_fingerprints.items():
        values = tuple(str(value) for value in observed.get(case_id, ()))
        path_values = tuple(str(value).replace("\\", "/") for value in paths.get(case_id, ()))
        if values != (expected_hash,) or path_values != (expected_path,):
            raise SystemExit(
                f"repair CSV source fingerprint mismatch for {case_id}: "
                f"expected ({expected_path}, {expected_hash}), observed ({path_values}, {values})"
            )


def merge(canonical: Path, repair: Path, output: Path, *, in_place: bool) -> None:
    if not canonical.is_file() or not repair.is_file():
        raise SystemExit("canonical and repair CSV paths must both exist")
    full = pd.read_csv(canonical, low_memory=False)
    repaired = pd.read_csv(repair, low_memory=False)
    _validate_repair(repaired)
    if list(full.columns) != list(repaired.columns):
        missing = sorted(set(full.columns) - set(repaired.columns))
        extra = sorted(set(repaired.columns) - set(full.columns))
        raise SystemExit(f"CSV schema mismatch: missing={missing}, extra={extra}")
    if full.duplicated(KEYS, keep=False).any():
        raise SystemExit("canonical CSV already contains duplicate protocol cells")
    before = len(full)
    old_cases = full[full["case_id"].isin(REPAIR_CASES)]
    if len(old_cases) != len(REPAIR_CASES) * EXPECTED_METHODS * len(EXPECTED_BUDGETS) * EXPECTED_REPEATS:
        raise SystemExit("canonical CSV does not contain exactly 2,400 historical rows per repair case")
    merged = pd.concat([full[~full["case_id"].isin(REPAIR_CASES)], repaired], ignore_index=True)
    if len(merged) != before:
        raise SystemExit(f"merged row count changed: before={before}, after={len(merged)}")
    if merged.duplicated(KEYS, keep=False).any():
        raise SystemExit("merged CSV contains duplicate protocol cells")
    merged = merged.sort_values(KEYS, kind="mergesort").reset_index(drop=True)
    output = output.resolve()
    if output == canonical.resolve() and not in_place:
        raise SystemExit("refusing in-place overwrite without --in-place")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=output.parent, suffix=".csv") as fh:
        tmp = Path(fh.name)
    try:
        merged.to_csv(tmp, index=False)
        os.replace(tmp, output)
    finally:
        if tmp.exists():
            tmp.unlink()
    print(f"wrote {output} ({len(merged)} rows)")
    print(f"sha256={_sha256(output)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=ROOT / "benchmark/results/full_benchmark/results.csv")
    parser.add_argument("--repair", type=Path, required=True, help="7-case repair CSV from the Linux run")
    parser.add_argument("--output", type=Path, required=True, help="replacement CSV path")
    parser.add_argument("--in-place", action="store_true", help="allow --output to equal --canonical")
    args = parser.parse_args()
    merge(args.canonical, args.repair, args.output, in_place=args.in_place)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
