"""
examples/01_quickstart.py
==========================
DOE Benchmark case library — minimal usage example.

This toolbox **provides simulation cases only** and does not include any
DOE methods themselves.  Use your DOE tool of choice (scipy.stats.qmc,
SALib, BoTorch, ...) to generate design points in the normalised [0,1]^d
space, then call `case.evaluate(X)` to obtain the physically dimensioned
simulation response.

How to run
----------
    python examples/01_quickstart.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import doe_testbed as db


# ── 1. Browse the case catalogue ────────────────────────────────────────────
print("=" * 60)
print(" DOE Benchmark case library — Quick start")
print("=" * 60)

all_cases = db.list_cases()
print(f"\nRegistered cases: {len(all_cases)}\n")

from collections import defaultdict
by_domain = defaultdict(list)
for m in all_cases:
    by_domain[m.domain].append(m.case_id)
for domain, ids in sorted(by_domain.items()):
    print(f"  [{domain:20s}]  {ids}")


# ── 2. Load a case and inspect its metadata ─────────────────────────────────
print("\n" + "─" * 60)
print(" Case: rectangular cantilever beam (cantilever_beam_5d)")
print("─" * 60)

case = db.get_case("cantilever_beam_5d")
m = case.meta

print(f"  Name              : {m.name}")
print(f"  Domain            : {m.domain}")
print(f"  Dimensions        : {m.dim_input}D → {m.dim_output}")
print(f"  Governing equation: {m.governing_equation[:60]}...")
print(f"  Solver            : {m.solver_library}")

print("\n  Input variables:")
for v in m.input_variables:
    print(f"    {v.symbol:>8}  {v.name:<12}  "
          f"[{v.lower:.3g}, {v.upper:.3g}] {v.unit}")

print("\n  Output variables:")
for v in m.output_variables:
    print(f"    {v.symbol:>8}  {v.name:<12}  unit={v.unit}")


# ── 3. Generate design points with your own DOE method, then call case.evaluate ──
print("\n" + "─" * 60)
print(" Generating LHS design points with scipy.stats.qmc and evaluating")
print("─" * 60)

from scipy.stats.qmc import LatinHypercube

sampler = LatinHypercube(d=case.dim, seed=42)
X_norm = sampler.random(20)          # normalised space, shape (20, 5)
Y = case.evaluate(X_norm)            # physically dimensioned response, shape (20, 3)

print(f"\n  X_norm.shape = {X_norm.shape}   (normalised coordinates [0,1]^d)")
print(f"  Y.shape      = {Y.shape}   "
      f"(outputs: {', '.join(m.output_names)})")

print(f"\n  First 5 samples:")
header = "  " + "  ".join(f"{v.name:>12}" for v in m.output_variables)
print(header)
for i in range(5):
    row = "  " + "  ".join(f"{Y[i, j]:>12.3f}"
                           for j in range(m.dim_output))
    print(row)


# ── 4. Convert between normalised and physical coordinates ──────────────────
print("\n" + "─" * 60)
print(" Normalised <-> physical coordinate conversion")
print("─" * 60)

X_phys = case.denormalize(X_norm)
print(f"  Sample 1 physical values:"
      f" F={X_phys[0,0]:.1f} kN,"
      f" L={X_phys[0,1]:.2f} m,"
      f" b={X_phys[0,2]*1000:.0f} mm,"
      f" h={X_phys[0,3]*1000:.0f} mm,"
      f" E={X_phys[0,4]:.0f} GPa")


# ── 5. Automated self-check ────────────────────────────────────────────────
print("\n" + "─" * 60)
print(" Case self-check (check_implementation)")
print("─" * 60)

result = case.check_implementation(verbose=False)
print(f"  {result['summary']}")

print("\nQuick-start example finished.")
