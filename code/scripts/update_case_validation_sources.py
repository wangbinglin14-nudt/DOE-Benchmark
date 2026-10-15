"""Generate or verify docs/case_validation_sources.csv.

The human case-audit ledger lives in ``docs/case_validation_status.csv``.
This companion CSV pins that ledger to the current registered case source
files, so a future response-surface edit cannot silently inherit an older
"accepted" audit disposition.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT = ROOT / "docs" / "case_validation_sources.csv"
FIELDNAMES = ["case_id", "case_source_path", "case_source_sha256"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def current_rows() -> list[dict[str, str]]:
    import importlib

    import doe_testbed as db

    rows: list[dict[str, str]] = []
    for meta in sorted(db.list_cases(), key=lambda item: item.case_id):
        case = db.get_case(meta.case_id)
        module = importlib.import_module(type(case).__module__)
        source = Path(str(module.__file__)).resolve()
        try:
            rel = source.relative_to(ROOT).as_posix()
        except ValueError:
            rel = source.as_posix()
        rows.append({
            "case_id": meta.case_id,
            "case_source_path": rel,
            "case_source_sha256": _sha256(source),
        })
    return rows


def _csv_text(rows: list[dict[str, str]]) -> str:
    import io

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
        help="case validation source fingerprint CSV to write or check.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Rewrite the output CSV. Without this flag, only verify it is current.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    expected = _csv_text(current_rows())
    output = args.output.resolve()

    if args.write:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(expected, encoding="utf-8")
        print(f"[ok] wrote {output}")
        return 0

    if not output.exists():
        print(f"[ERROR] missing case validation source CSV: {output}", file=sys.stderr)
        return 1
    observed = output.read_text(encoding="utf-8")
    if observed != expected:
        print(
            f"[ERROR] {output} is stale; run "
            f"python scripts/update_case_validation_sources.py --write",
            file=sys.stderr,
        )
        return 1
    print(f"[ok] {output} is current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
