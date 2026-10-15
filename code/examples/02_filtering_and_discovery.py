"""
examples/02_filtering_and_discovery.py
========================================
Demonstrates the case registry query API: filtering by domain, dimension,
cost level, VS scenario, and other metadata attributes.

How to run
----------
    python examples/02_filtering_and_discovery.py
"""

import warnings
warnings.filterwarnings("ignore")

import doe_testbed as db
from doe_testbed.registry.registry import get_minimum_covering_set, print_cases_table


# ── 1. List all cases ────────────────────────────────────────────────────────
print("=" * 60)
print(" Case Discovery & Filtering")
print("=" * 60)

all_ids = db.list_cases(return_ids=True)
print(f"\nTotal registered cases: {len(all_ids)}")


# ── 2. Filter by domain ──────────────────────────────────────────────────────
print("\n--- Aerodynamics cases ---")
aero = db.list_cases(domain="aerodynamics")
for m in aero:
    print(f"  {m.case_id:<30} {m.dim_input}D→{m.dim_output}  {m.cost_level}")


# ── 3. Filter by dimension range ─────────────────────────────────────────────
print("\n--- High-dimensional cases (d ≥ 8) ---")
high_d = db.list_cases(dim_input_range=(8, 100))
for m in high_d:
    print(f"  {m.case_id:<30} {m.dim_input}D")


# ── 4. Filter by VS scenario ─────────────────────────────────────────────────
print("\n--- Cases supporting reliability analysis (VS-G) ---")
vs_g = db.list_cases(applicable_scenarios=["VS-G"])
for m in vs_g:
    print(f"  {m.case_id:<30} failure_boundary={m.has_failure_boundary}")


# ── 5. Filter by cost level ──────────────────────────────────────────────────
print("\n--- Cheap cases (L0, < 1 ms) ---")
cheap = db.list_cases(cost_level="L0", max_eval_time_ms=1.0)
for m in cheap:
    t = m.typical_eval_time_ms.get("default", 0)
    print(f"  {m.case_id:<30} {t:.2f} ms")


# ── 6. Multi-fidelity cases ──────────────────────────────────────────────────
print("\n--- Multi-fidelity cases ---")
mf = db.list_cases(supports_multifidelity=True)
for m in mf:
    print(f"  {m.case_id:<30} fidelity_levels={m.fidelity_levels}")


# ── 7. Minimum covering set (all 8 VS scenarios) ─────────────────────────────
print("\n--- Minimum covering set (spans all 8 VS scenarios) ---")
covering = get_minimum_covering_set()
for c in covering:
    print(f"  {c.meta.case_id:<30} scenarios={c.meta.applicable_scenarios}")
print(f"\n  → {len(covering)} cases cover all 8 VS scenarios")


# ── 8. Print formatted table ─────────────────────────────────────────────────
print("\n--- Full case table (structures domain only) ---")
structs = db.list_cases(domain="structures")
print_cases_table(structs)

print("\nFiltering example finished.")
