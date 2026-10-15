"""
examples/03_sobol_and_sensitivity.py
========================================
Demonstrates how to access precomputed Sobol indices and use them
for variable screening, and how to compute your own Sobol indices
using SALib with doe-testbed cases.

How to run
----------
    python examples/03_sobol_and_sensitivity.py

Requirements
------------
    pip install SALib   (for on-the-fly computation; precomputed indices need no extra deps)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import doe_testbed as db


# ── 1. Access precomputed Sobol indices ──────────────────────────────────────
print("=" * 60)
print(" Precomputed Sobol Indices")
print("=" * 60)

case = db.get_case("cantilever_beam_5d")
m = case.meta

print(f"\nCase: {m.case_id}")
print(f"Sobol computation: {m.sobol_computation}")
print(f"\n{'Variable':<12} {'S1':>8} {'ST':>8}")
print("-" * 30)
for var in m.input_variables:
    s1 = m.sobol_S1.get(var.symbol, float("nan"))
    st = m.sobol_ST.get(var.symbol, float("nan"))
    print(f"{var.symbol:<12} {s1:>8.3f} {st:>8.3f}")


# ── 2. Variable screening using ST ──────────────────────────────────────────
print(f"\n--- Variable screening (ST > 0.05) ---")
print(f"  Effective dimension: {m.effective_dim}")
print(f"  Important variables: {m.important_variables}")

active = {k: v for k, v in m.sobol_ST.items() if v >= 0.05}
inactive = {k: v for k, v in m.sobol_ST.items() if v < 0.05}
print(f"  Active (ST≥0.05):   {sorted(active, key=active.get, reverse=True)}")
if inactive:
    print(f"  Inactive (ST<0.05): {sorted(inactive, key=inactive.get, reverse=True)}")


# ── 3. Interaction ratio ────────────────────────────────────────────────────
print(f"\n--- Interaction effects ---")
sum_s1 = sum(m.sobol_S1.values())
sum_st = sum(m.sobol_ST.values())
ir = 1 - sum_s1 / sum_st if sum_st > 0 else 0
print(f"  Sum(S1) = {sum_s1:.3f}")
print(f"  Sum(ST) = {sum_st:.3f}")
print(f"  Interaction ratio = 1 - Sum(S1)/Sum(ST) = {ir:.3f}")
if ir > 0.3:
    print("  → Strong interactions: additive models may lose accuracy")
elif ir > 0.1:
    print("  → Moderate interactions: consider second-order terms")
else:
    print("  → Weak interactions: first-order models likely sufficient")


# ── 4. Compare Sobol across multiple cases ───────────────────────────────────
print(f"\n{'='*60}")
print(f" Sobol Summary Across All 34 Cases")
print(f"{'='*60}")

print(f"\n{'Case ID':<35} {'d':>3} {'eff_d':>5} {'max_ST':>7} {'IR':>6}")
print("-" * 60)
for meta in db.list_cases():
    if meta.sobol_ST is None:
        continue
    max_st = max(meta.sobol_ST.values())
    sum_s1 = sum(meta.sobol_S1.values())
    sum_st = sum(meta.sobol_ST.values())
    ir = 1 - sum_s1 / sum_st if sum_st > 0 else 0
    print(f"{meta.case_id:<35} {meta.dim_input:>3} {meta.effective_dim:>5} "
          f"{max_st:>7.3f} {ir:>6.3f}")


# ── 5. (Optional) Compute Sobol from scratch with SALib ─────────────────────
print(f"\n{'='*60}")
print(f" Computing Sobol from scratch (cantilever_beam_5d, n=512)")
print(f"{'='*60}")

try:
    from SALib.sample import saltelli
    from SALib.analyze import sobol as sobol_analyze

    problem = {
        "num_vars": case.dim,
        "names": [v.symbol for v in m.input_variables],
        "bounds": [[0.0, 1.0]] * case.dim,
    }

    X_sa = saltelli.sample(problem, 512)
    Y_sa = case.evaluate(X_sa)[:, 0]  # primary output

    si = sobol_analyze.analyze(problem, Y_sa)
    print(f"\n{'Variable':<12} {'S1':>8} {'ST':>8}  {'S1(pre)':>8} {'ST(pre)':>8}")
    print("-" * 50)
    for i, name in enumerate(problem["names"]):
        print(f"{name:<12} {si['S1'][i]:>8.3f} {si['ST'][i]:>8.3f}"
              f"  {m.sobol_S1[name]:>8.3f} {m.sobol_ST[name]:>8.3f}")

    print("\n  (pre) = precomputed values from case metadata")

except ImportError:
    print("\n  SALib not installed; skipping on-the-fly computation.")
    print("  Install with: pip install SALib")

print("\nSensitivity analysis example finished.")
