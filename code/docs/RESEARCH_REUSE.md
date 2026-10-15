# Research reuse of the current workspace

## What this workspace contains

This is a computational case-study benchmark: 34 registered engineering cases,
20 headline methods, four budgets and 30 repeats (81,600 archived cells).
These are simulator-derived data, not raw physical measurements. Source hashes
and rerendering establish internal traceability, not independent authentication
of the historical run or experimental validity of the physical assumptions.

The public `code/` tree contains the research implementation in `doe_testbed/`,
`benchmark/`, `scripts/`, `tools/`, `tests/` and `examples/`, with build
metadata and documentation; figure data, tables and protocols are in the
repository-root `manuscripts/` directory. This includes case
models, analytical comparators, all method implementations (including methods
outside the headline pool), metrics, analysis, figure generation and tests.

## Locate the current data

Work from the repository-root `code/` for code and raw results, and from
`manuscripts/` for
figures, tables, protocols and derived figure data. The canonical
result is `benchmark/results/full_benchmark/results.csv`, with SHA-256
`71e72d937b64abc6f5c44da2f6b2183e293e228258ce4c0abce9bb4100f0ca0b`.
Check the run manifest before interpreting derived results. The public package
scope is described in the [repository README](../../README.md).

## Install and explore

Use an isolated Python environment. The archived `run_config.json` records the
formal startup versions. `requirements.lock` and
`python_environment_pip_freeze.txt` provide dependency records, but do not
constitute a complete reconstruction of the original host. The later packaging
environment is distinct from the original run environment. Check Python and
Torch compatibility before installing the benchmark extra; its Torch constraint
is `>=2.2,<2.5` and is not supported by every newer Python release.

```sh
python -m pip install -e .
python -c "import doe_testbed as db; print(db.list_cases())"
python -m pip install -e ".[benchmark]"
python -m pytest tests benchmark/tests -q
```

For individual cases, install the corresponding optional dependencies only when
needed (`aero`, `chem`). Read the case's evidence note before selecting fidelity
or interpreting failure outputs. The nominal `full` extra excludes Cantera.
Archived headline results must not be mixed with newly computed outputs using
a different backend.

## Inspect the evidence before extending a case

[CASE_EVIDENCE_INDEX.md](CASE_EVIDENCE_INDEX.md) links one note per case,
stored beside its Python implementation. A note records equations and units,
parameter provenance, solver behavior, targeted tests and limitations. Original
source evidence means a traceable original paper/book/report passage, equation,
table or dataset record; a bibliographic match alone does not verify a coefficient.
Unavailable source passages remain explicitly unresolved. Cases with different
dimensions sharing one Python module still have individual notes.

Computational `.py` sources are preserved byte-for-byte because the historical
CSV and manifest pin those bytes. New `.evidence.md` documents disclose legacy
comment errors without falsely assigning historical data to modified code.
For corrected physics or changed parameter ranges, create a new version and
fresh results. See [CONTRIBUTING.md](project/CONTRIBUTING.md).

## Reproduce existing figures and numbers

The following historical commands ran from the development code directory in a
disposable workspace. Some retain `paper/` defaults; use explicit path options
for `manuscripts/` where supported, as described in the [repository README](../../README.md):

```sh
python scripts/prepare_figdata.py --require-headline-source
python scripts/plots_nature.py
python scripts/fill_paper_numbers.py --require-results-csv --require-results-contract --require-headline-source --require-submission-complete
python scripts/check_submission_ready.py paper
```

These commands reproduce from retained derived artifacts and the frozen raw
archive. To recompute D2 and D3 from the raw CSV as well:

```sh
python benchmark/scripts/run_sample_complexity.py benchmark/results/full_benchmark/results.csv --threshold 0.20 --output-dir benchmark/results/sample_complexity_0.20 --require-headline-source
python benchmark/scripts/run_reliability_recovery.py benchmark/results/full_benchmark/results.csv --budget-mult 10 --output-dir benchmark/results/reliability_recovery --require-headline-source
```

Run these in a disposable copy: they intentionally regenerate
derived files. [FIGURE_DATA_TRACEABILITY.md](FIGURE_DATA_TRACEABILITY.md)
maps all six figures and 14 supplementary tables to their inputs and generators.
The original full sweep is a separate, expensive experiment. Follow
[REPRODUCIBILITY.md](REPRODUCIBILITY.md)
for its preflight and server protocol. Do not use an unverified historical clone
URL as evidence that a public deposit exists.

## Licence, citation and public deposition

The project code and existing software-deposit metadata declare MIT; retain
`LICENSE` and the third-party notices. Third-party papers and books retain their
own rights; the case notes cite them instead of redistributing their full text.
No new licence for third-party material is granted by this workspace. Confirm a
data-specific licence and author-owned document permissions when publishing the
archive. `CITATION.cff` supplies citation metadata for the software; the
manuscript is prepared for submission, not claimed to be an accepted article.

The public repository is linked from the [repository README](../../README.md).
Repository availability does not establish an archival DOI or a prior independent
protocol-registration timestamp.
