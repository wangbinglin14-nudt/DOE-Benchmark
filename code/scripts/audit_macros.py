"""Static audit: every \\NumXxx macro used in the manuscripts must be
backed by either ``paper/numbers.tex`` or by an inline stub default in
the manuscript itself.  Run this before submission or in CI as a
fail-fast guard against undefined-macro LaTeX errors.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Set

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.project_layout import MANUSCRIPTS_DIR  # noqa: E402

PAPER = MANUSCRIPTS_DIR

USE_RE = re.compile(r"\\Num[A-Za-z0-9]+")
DEF_RE = re.compile(r"\\newcommand\{\\(Num[A-Za-z0-9]+)\}")

# Macros defined in main.tex with a non-trivial body (e.g. the
# ``\NumShiftCIAvailableTF`` helper) that aren't \newcommand-style.
HELPERS = {"\\NumShiftCIAvailableTF"}


def macros_used(p: Path) -> Set[str]:
    return set(USE_RE.findall(p.read_text(encoding="utf-8")))


def macros_defined(p: Path) -> Set[str]:
    return {f"\\{m}" for m in DEF_RE.findall(p.read_text(encoding="utf-8"))}


def audit(manuscript: Path, numbers: Path) -> int:
    used = macros_used(manuscript)
    defined_numbers = macros_defined(numbers) if numbers.exists() else set()
    defined_inline = macros_defined(manuscript)
    missing = used - defined_numbers - defined_inline - HELPERS
    print(f"\n[{manuscript.name}]")
    print(f"  uses {len(used)} \\Num macros")
    print(f"  numbers.tex defines {len(defined_numbers)}")
    print(f"  inline stubs       {len(defined_inline)}")
    if missing:
        print(f"  MISSING ({len(missing)}):")
        for m in sorted(missing):
            print(f"    {m}")
        return 1
    print("  ok - every macro is defined")
    return 0


def main() -> int:
    numbers = PAPER / "numbers.tex"
    rc = 0
    for name in ("nature_cs_main.tex", "preprint_arxiv.tex"):
        manuscript = PAPER / name
        if not manuscript.exists():
            print(f"\n[{name}]")
            print("  skipped - manuscript does not exist")
            continue
        rc |= audit(manuscript, numbers)
    return rc


if __name__ == "__main__":
    sys.exit(main())
