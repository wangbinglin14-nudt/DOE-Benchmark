# D1 Ranking-Reversal Directory

The current headline D1 analysis is the set of machine-readable artifacts in
this directory (`correlation.json`, the three pairing subdirectories, and
`d1_loo_sensitivity.csv`). It is generated from the formal 34-case,
20-method, 30-repeat sweep and is summarized in
`benchmark/results/full_benchmark/report/REPORT.md`.

The analytical anchor is the standalone six-function phase. Its physical rows
are selected from the canonical full-sweep CSV, not from the historical pilot
snapshot. The 19-row RSS endpoint repair is metadata-only; the complete audit
is recorded in `analytical_runtime_trace_repair_audit.json`. The pre-canonical
D1 files are retained under `_archive_20260831_d1_precanonical/` and are not
used for manuscript numbers.

`REPORT_legacy_pilot_20260430.md` is retained only as historical provenance.
It describes the earlier 1,200-run pilot (10 methods, 6 analytical cases and
6 physical cases) and must not be used for headline claims or combined with
the formal sweep.
