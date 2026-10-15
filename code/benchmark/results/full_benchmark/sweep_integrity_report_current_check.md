# Sweep Integrity Report

**Input**: `benchmark\results\full_benchmark\results.csv`

## Summary

- Rows: 81600
- Cases observed: 34
- Methods observed: 20
- Budgets observed: [5, 10, 20, 50]
- Status counts: `{'ok': 78605, 'timeout': 2977, 'failed': 18}`
- Failure-category counts: `{'ok': 78605, 'timeout': 2977, 'exception': 18}`
- OK fraction: 0.963
- Timeout fraction: 0.036


## Run Config Binding

- run_config.json contract matched partial results: benchmark\results\full_benchmark\run_config.json
- run_config.json runtime controls recorded and valid: benchmark\results\full_benchmark\run_config.json
- run_config.json runtime environment recorded and valid: benchmark\results\full_benchmark\run_config.json
- run_config.json torch reproducibility policy recorded and valid: benchmark\results\full_benchmark\run_config.json


## Findings

PASS: sweep satisfies the declared integrity gate.
