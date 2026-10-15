"""Audit effective_dim metadata against computed Sobol ST indices.

For each case, effective_dim should roughly equal the number of variables
with ST >= 0.05 for the primary output. Flags mismatches.
"""

import doe_testbed as db

ST_THRESHOLD = 0.05
issues = []

for cid in sorted(db.list_cases(return_ids=True)):
    case = db.get_case(cid)
    m = case.meta
    if m.sobol_ST is None:
        continue
    n_active = sum(1 for v in m.sobol_ST.values() if v >= ST_THRESHOLD)
    declared = m.effective_dim
    if abs(n_active - declared) > 1:
        issues.append((cid, declared, n_active, dict(m.sobol_ST)))

if issues:
    print(f"\n{len(issues)} effective_dim mismatches (|declared - active| > 1):\n")
    for cid, decl, act, st in issues:
        print(f"  {cid}: declared={decl}, active(ST>=0.05)={act}")
        top = sorted(st.items(), key=lambda x: -x[1])[:5]
        for k, v in top:
            print(f"    {k}: ST={v:.3f}")
        print()
else:
    print("All effective_dim values are consistent with Sobol ST.")
