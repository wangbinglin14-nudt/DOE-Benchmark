# `doe_testbed/` — Engineering case library

The pure case-library layer of DOE-Benchmark.  Provides 34
physics-grounded engineering test problems under a unified
`case.evaluate(X)` interface.  This package **does not** know about
DOE methods, surrogates, or Bayesian optimisation — those live in
[`../benchmark/`](../benchmark).

> **Frozen at `v0.4.0-prereg` (2026-05-01).**  See
> [`../paper/preregistration.md`](../../manuscripts/protocol/preregistration.md) §3.
> Adding a case after the freeze requires a
> `paper/preregistration_amendments.md` entry.

## 1. Layout

```
doe_testbed/
├── __init__.py                Public API: list_cases / get_case / get_case_meta
├── cli.py                     Console entry-point (`python -m doe_testbed`)
├── base/                      Abstract types: Case, CaseMeta, OutputVariable
├── cases/                     34 case modules (one per physics problem)
├── registry/                  Singleton registry + name resolution
└── utils/                     Sampling helpers (Sobol, LHS, normalisation)
```

| Subdirectory | What's inside |
|---|---|
| [`base/`](base) | The abstract base classes every case implements: `Case` (with `meta`, `evaluate`, optional `denormalize`), `CaseMeta` (the static problem description), `OutputVariable` (per-output bounds + optional `failure_threshold`). |
| [`cases/`](cases) | 34 + 3 case modules covering structures (8), heat transfer (7), aerodynamics (6), chemical processes (5), multidisciplinary design (4), geomechanics (4).  Three modules implement the `_5d/_10d/_20d` (variable-beam) and `_4d/_8d/_16d` (multilayer-wall) dimension-scalable families. |
| [`registry/`](registry) | Built-in case registry (eager imports + `__all__` list) plus the `list_cases()` / `get_case(id)` / `get_case_meta(id)` lookup functions used by both `benchmark/` and the test suite. |
| [`utils/`](utils) | Sampling helpers (`sobol_sample`, `lhs_sample`, `halton_sample`, `optimised_lhs_sample`) and normalise / denormalise utilities.  Mirror the `scipy.stats.qmc` API but with deterministic-seed semantics that match `BenchmarkRunner`. |

## 2. The 5-line public API

```python
import doe_testbed as db

print(len(db.list_cases()))           # 34 cases at v0.4.0-prereg
case = db.get_case("cantilever_beam_5d")
print(case.meta.dim_input)             # 5
X = case.sample(50, method="lhs", seed=42)   # (50, 5) in [0, 1]^5
Y = case.evaluate(X)                   # (50, 3) physically dimensioned
```

The full quickstart is in the [repository README](../../README.md#quick-start).

## 3. CaseMeta

Every case carries a `CaseMeta` instance accessible as `case.meta`:

| Field | Type | Notes |
|---|---|---|
| `case_id` | `str` | Unique identifier; matches the module name. |
| `dim_input` | `int` | Number of input variables. |
| `dim_output` | `int` | Usually 3 across the suite. |
| `domain` | `str` | One of `"structures" / "heat_transfer" / "aerodynamics" / "chemical" / "multidisciplinary" / "geomechanics"`. |
| `applicable_scenarios` | `tuple[str, ...]` | "VS" tags (`VS-A` … `VS-H`). |
| `response_topology` | `tuple[str, ...]` | Closed vocabulary: `smooth_unimodal / smooth_multimodal / smooth_monotone / strong_interaction / sparse_effects / discontinuous / threshold_jump / heterogeneous`. |
| `cost_level` | `str` | `"L0"` (cheapest) up to `"L1+"` (heavier). |
| `output_variables` | `list[OutputVariable]` | Per-output bounds + optional `failure_threshold` + optional `failure_direction`. |
| `has_failure_boundary` | `bool` | Convenience flag — true iff any output declares a `failure_threshold`. |
| `supports_multifidelity` | `bool` | Activates the A2 amendment's MF sweep. |

## 4. Adding a new case

1. Drop a new module in [`cases/`](cases) implementing the `Case` ABC.
2. Register it in [`registry/`](registry).
3. Add a row to [`tests/test_cases.py`](../tests/test_cases.py) pinning
   shape + bounds + response-feature labels + (optional) failure threshold.
4. Update [`../paper/preregistration.md`](../../manuscripts/protocol/preregistration.md)
   only via [`../paper/preregistration_amendments.md`](../../manuscripts/protocol/preregistration_amendments.md)
   if the project is past freeze.
5. CI runs `tests/test_metadata.py` to verify `response_topology` / domain tags
   against the closed vocabularies.

## 5. CLI

```bash
python -m doe_testbed list                       # List all 34 case_ids
python -m doe_testbed describe cantilever_beam_5d   # Print CaseMeta
python -m doe_testbed evaluate cantilever_beam_5d --n 20 --method sobol --seed 42
```

See [`cli.py`](cli.py) for the argparse layer.

## 6. Cross-references

- [Repository README](../../README.md) — project overview.
- [`../ARCHITECTURE.md`](../docs/overview/ARCHITECTURE.md) §2.1 — library-layer
  rules.
- [`../tests/README.md`](../tests/README.md) — case-library test
  layout.
- [`../paper/preregistration.md`](../../manuscripts/protocol/preregistration.md) §3 —
  frozen case registry.
- [Case overview](../docs/CASE_OVERVIEW.md) — human-readable case
  catalogue; `CaseMeta` objects under `doe_testbed/cases/` are the
  machine-readable source of truth.
