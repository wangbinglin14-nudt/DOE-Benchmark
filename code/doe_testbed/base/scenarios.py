"""
doe_testbed.base.scenarios
============================
VS-System validation scenario constants and validation utilities.

The VS-System defines 8 standard validation scenarios for DOE method
evaluation (see design document, Section 9):

  VS-A  Space-filling quality
  VS-B  Effect estimation
  VS-C  Surrogate accuracy
  VS-D  Small-sample efficiency
  VS-E  High-dimensional adaptability
  VS-F  Variable screening
  VS-G  Sequential / adaptive design
  VS-H  Robust (Taguchi) design

This module provides:
  * String constants for each scenario (to avoid typos)
  * ``VS_ALL`` tuple of all valid scenarios
  * ``validate_scenarios()`` for checking arbitrary scenario lists
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

# ── Individual scenario constants ─────────────────────────────────────────
VS_A: str = "VS-A"
"""Space-filling quality (uniformity / low discrepancy)"""

VS_B: str = "VS-B"
"""Effect estimation (main effects / interactions, unconfounded)"""

VS_C: str = "VS-C"
"""Surrogate accuracy (global prediction RMSE / Q²)"""

VS_D: str = "VS-D"
"""Small-sample efficiency (reliability when N << conventional)"""

VS_E: str = "VS-E"
"""High-dimensional adaptability (performance decay d: 5→20→50)"""

VS_F: str = "VS-F"
"""Variable screening (identify important factors from many)"""

VS_G: str = "VS-G"
"""Sequential / adaptive design (modelling / optimisation / reliability)"""

VS_H: str = "VS-H"
"""Robust (Taguchi) design (stability under noise factors)"""

# ── Complete scenario set ─────────────────────────────────────────────────
VS_ALL: Tuple[str, ...] = (VS_A, VS_B, VS_C, VS_D, VS_E, VS_F, VS_G, VS_H)
"""Immutable tuple of all valid VS scenarios."""

# ── Scenario descriptions (for describe() / docs) ──────────────────────
VS_DESCRIPTIONS = {
    VS_A: "Space-filling quality (uniformity / low discrepancy)",
    VS_B: "Effect estimation (main + interaction effects, unconfounded)",
    VS_C: "Surrogate accuracy (global prediction RMSE / Q²)",
    VS_D: "Small-sample efficiency (N << conventional requirement)",
    VS_E: "High-dimensional adaptability (d: 5→20→50)",
    VS_F: "Variable screening (identify important factors)",
    VS_G: "Sequential / adaptive design (modelling / optimisation / reliability)",
    VS_H: "Robust (Taguchi) design (stability under noise factors)",
}


# ── Validation ───────────────────────────────────────────────────────────
def validate_scenarios(scenarios: Iterable[str]) -> List[str]:
    """Validate a scenario list against VS_ALL.

    Parameters
    ----------
    scenarios : iterable of str
        Scenario identifiers to validate, e.g. ["VS-A", "VS-G"].

    Returns
    -------
    list of str
        Deduplicated, order-preserved list of valid scenarios.

    Raises
    ------
    ValueError
        If any entry is not in VS_ALL.
    """
    seen = []
    bad = []
    for s in scenarios:
        if s not in VS_ALL:
            bad.append(s)
        elif s not in seen:
            seen.append(s)
    if bad:
        raise ValueError(
            f"applicable_scenarios contains invalid entries {bad}; "
            f"valid values are {list(VS_ALL)}"
        )
    return seen


__all__ = [
    "VS_A",
    "VS_B",
    "VS_C",
    "VS_D",
    "VS_E",
    "VS_F",
    "VS_G",
    "VS_H",
    "VS_ALL",
    "VS_DESCRIPTIONS",
    "validate_scenarios",
]
