# Benchmark results

This directory contains generated benchmark outputs. Run commands from the
repository root and keep each run in its named subdirectory.

| Directory | Role | Current use |
|---|---|---|
| `full_benchmark/` | Canonical 34-case x 20-method formal sweep and report. | Source for manuscript numbers and submission artifacts. |
| `ranking_reversal/` | D1 analytical-versus-physics ranking analysis. | Canonical D1 analysis input. |
| `sample_complexity_0.20/` | D2 topology/sample-complexity analysis. | Canonical D2 analysis input. |
| `reliability_recovery/` | D3 failure-probability recovery analysis. | Canonical D3 analysis input. |
| `sensitivity_recovery/` | Supplementary Sobol recovery analysis. | Supplemental result; regenerate when needed. |
| `_smoke/`, `_smoke_gate/` | Bounded local pipeline checks. | Validation only; never merge into the formal sweep. |
| `provenance_repair_20260831/`, `reprovenance_rerun/` | Dated repair and provenance reruns. | Historical diagnostics. |

Large generated outputs are excluded from normal source review through the
repository ignore configuration. The formal result contract is documented in
`docs/REPRODUCIBILITY.md` and `docs/PROJECT_ARCHIVE_INDEX_20260909.md`.
