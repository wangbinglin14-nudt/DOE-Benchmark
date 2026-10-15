"""Generate or verify docs/case_validation_note_sources.csv.

The case-validation ledger references human review notes and supporting
reference files under ``案例正确性校验/``.  This companion CSV pins those files
by path, size, and SHA-256 so a later attachment replacement cannot silently
inherit an older validation summary.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
VALIDATION_ROOT = ROOT / "案例正确性校验"
DEFAULT_OUTPUT = ROOT / "docs" / "case_validation_note_sources.csv"
FIELDNAMES = ["path", "size_bytes", "sha256"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def current_rows() -> list[dict[str, str]]:
    if not VALIDATION_ROOT.exists():
        return []
    rows: list[dict[str, str]] = []
    for path in sorted(VALIDATION_ROOT.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.name.startswith("~$"):
            continue
        rel = path.relative_to(ROOT).as_posix()
        rows.append({
            "path": rel,
            "size_bytes": str(path.stat().st_size),
            "sha256": _sha256(path),
        })
    return rows


def csv_text(rows: list[dict[str, str]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="case validation note/reference fingerprint CSV to write or check.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Rewrite the output CSV. Without this flag, only verify it is current.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    expected = csv_text(current_rows())
    output = args.output.resolve()

    if args.write:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(expected, encoding="utf-8")
        print(f"[ok] wrote {output}")
        return 0

    if not output.exists():
        print(f"[ERROR] missing case validation note-source CSV: {output}", file=sys.stderr)
        return 1
    observed = output.read_text(encoding="utf-8")
    if observed != expected:
        print(
            f"[ERROR] {output} is stale; run "
            f"python scripts/update_case_validation_note_sources.py --write",
            file=sys.stderr,
        )
        return 1
    print(f"[ok] {output} is current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
