# `tests/` — Case-library integration tests

This directory holds the integration-level tests for the
`doe_testbed/` case library (the physics layer).  Method-side tests
for `benchmark/` live separately under
[`../benchmark/tests/`](../benchmark/tests) — see §2 below for the
split.

## 1. Layout

| File | Role | Tests |
|---|---|---|
| [`test_cases.py`](test_cases.py) | Per-case consistency: `meta.dim_input` matches `evaluate(X)` shape; outputs respect declared bounds; `failure_threshold` semantics. | ~25 |
| [`test_metadata.py`](test_metadata.py) | Registry-level invariants: every case has a `CaseMeta`; tags conform to the closed vocabulary; output-variable count matches `dim_output`. | ~15 |
| [`test_utils.py`](test_utils.py) | Sampling utilities: Sobol' / LHS shape + reproducibility; normalise / denormalise round-trip. | ~20 |
| [`_audit_physics.py`](_audit_physics.py) | Manual audit helpers (not a pytest module — leading underscore); used to author the `_cross_validation/` write-ups. | n/a |
| [`_cross_validation/`](_cross_validation) | Per-case cross-validation write-ups (Markdown) — historical artefacts checked into git so the physical-correctness audit is auditable; the live tests are in `test_cases.py`. | n/a |

## 2. The two-tier test layout

The project has two test directories with **different scopes**:

| Directory | Scope | Approximate count (2026-05-23) |
|---|---|---|
| `tests/` | `doe_testbed` library — physics, registry, sampling utilities. | ~50 (5 files) |
| `benchmark/tests/` | `benchmark` package — method adapters, runner, statistics, paper pipeline. | 281 (19 files) |

The split mirrors the architectural rule that `doe_testbed/` knows
nothing about `benchmark/` (see [`../ARCHITECTURE.md`](../docs/overview/ARCHITECTURE.md)
§1).  The library tests can run without any DOE/surrogate
dependencies; the benchmark tests pull in `scikit-learn`, `scipy`,
optional `botorch`/`pyro`, etc.

## 3. Running

```bash
# Both tiers (the default for CI):
python -m pytest tests benchmark/tests

# Just the library tier (fast, < 30 s):
python -m pytest tests

# Just the benchmark tier (slower, ~90 s; pulls in BoTorch if installed):
python -m pytest benchmark/tests

# A single test:
python -m pytest tests/test_cases.py::test_cantilever_beam_5d_outputs_within_bounds -v
```

The default `pyproject.toml::[tool.pytest.ini_options]` collects only
`tests/`; `benchmark/tests/` is collected explicitly so a developer
can split fast / slow CI tiers.

## 4. Adding a new test

| Where | When |
|---|---|
| `tests/` | Adding a new case, new sampling utility, new physics constraint. |
| `benchmark/tests/` | Adding a new method, new statistic, new paper-pipeline contract. |

Naming convention: `test_<module>_<concern>.py`.  Pin every numeric
contract that downstream paper macros depend on.

## 5. Cross-references

- [`../benchmark/tests/`](../benchmark/tests) — benchmark-side tests.
- [`../benchmark/README.md`](../benchmark/README.md) §"Test suite" —
  per-file index.
- [`../docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md) §13 —
  CI commands run on every PR.
