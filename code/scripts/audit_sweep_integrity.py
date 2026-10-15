"""Audit a DOE-Benchmark sweep ``results.csv`` before paper stages run.

The runner is intentionally resumable and tolerant of per-cell failures.
That is useful operationally, but paper artifacts need a stricter gate:
the CSV must contain the declared case/method/budget/repeat grid, enough
completed repeats, and an acceptable success rate before Stage 3-10 outputs
are treated as authoritative.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import doe_testbed as db
from benchmark.runtime_contracts import (
    ALLOWED_FAILURE_CATEGORIES,
    RESULT_SCHEMA_VERSION,
    csv_header_contract_errors,
    failure_category_is_consistent,
    invalid_protocol_digest_mask,
    invalid_runtime_duration_mask,
    strict_integer_series,
)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


REQUIRED_COLUMNS = {
    "case_id",
    "case_source_path",
    "case_source_sha256",
    "domain",
    "dim",
    "method",
    "method_family",
    "method_source_path",
    "method_source_sha256",
    "budget_mult",
    "n_train",
    "n_test",
    "repeat",
    "result_schema_version",
    "status",
    "failure_category",
    "t_total_s",
    "test_seed",
    "error",
    "n_output",
    "nrmse",
    "seed",
    "seed_strategy",
    "seed_start",
    "protocol_digest",
    "worker_host",
    "worker_pid",
    "cell_started_utc",
    "cell_finished_utc",
    "rss_gb_start",
    "rss_gb_end",
    "rss_gb_peak",
}
ALLOWED_STATUS = {"ok", "timeout", "failed"}
CELL_KEY_COLUMNS = ["case_id", "method", "budget_mult", "repeat"]
OPTIONAL_OK_NUMERIC_METRICS = ["rmse", "mae", "maxae", "q95ae", "r2"]
RUNTIME_TRACE_COLUMNS = [
    "worker_host",
    "worker_pid",
    "cell_started_utc",
    "cell_finished_utc",
    "rss_gb_start",
    "rss_gb_end",
    "rss_gb_peak",
]


def _current_case_source(case_id: str) -> tuple[str | None, str | None, str | None]:
    try:
        case = db.get_case(case_id)
        module = __import__(type(case).__module__, fromlist=["__file__"])
        source = Path(str(module.__file__)).resolve()
        h = hashlib.sha256()
        with source.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        try:
            rel = source.relative_to(ROOT).as_posix()
        except ValueError:
            rel = source.as_posix()
        return rel, h.hexdigest(), None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def _current_method_sources(pool_methods: Sequence[str]) -> dict[str, dict[str, str]]:
    try:
        from benchmark.core import method_source_fingerprint
        from benchmark.scripts.run_full_benchmark import POOLS
    except Exception:
        return {}

    out: dict[str, dict[str, str]] = {}
    for pool_fn in POOLS.values():
        try:
            methods = pool_fn()
        except Exception:
            continue
        for method in methods:
            if method.name in out:
                continue
            if pool_methods and method.name not in pool_methods:
                continue
            try:
                out[method.name] = method_source_fingerprint(method)
            except Exception:
                continue
    return out


def _normalize_source_path(value: object) -> str:
    text = str(value).strip().replace("\\", "/")
    if not text:
        return ""
    try:
        path = Path(text)
        if path.is_absolute():
            try:
                return path.resolve().relative_to(ROOT).as_posix()
            except (OSError, ValueError):
                return path.as_posix()
    except (OSError, ValueError):
        pass
    return text


def _source_paths_match(observed_paths: Sequence[str], expected_path: str | None) -> bool:
    expected = _normalize_source_path(expected_path)
    if not observed_paths or not expected:
        return False
    return all(
        observed == expected or observed.endswith(f"/{expected}")
        for observed in observed_paths
    )


def _split_csv_arg(value: str | None) -> list[str] | None:
    if value is None:
        return None
    out = [x.strip() for x in value.replace(";", ",").split(",") if x.strip()]
    return out or None


def _split_mapping_arg(value: str | None) -> dict[str, str] | None:
    items = _split_csv_arg(value)
    if not items:
        return None
    out: dict[str, str] = {}
    for item in items:
        if ":" not in item:
            raise ValueError(f"mapping item must be name:value, got {item!r}")
        key, val = item.split(":", 1)
        key = key.strip()
        val = val.strip()
        if key:
            out[key] = val
    return out or None


def _status_counts(df: pd.DataFrame) -> dict[str, int]:
    if "status" not in df.columns:
        return {}
    return {str(k): int(v) for k, v in df["status"].value_counts(dropna=False).items()}


def _failure_category_counts(df: pd.DataFrame) -> dict[str, int]:
    if "failure_category" not in df.columns:
        return {}
    return {
        str(k): int(v)
        for k, v in df["failure_category"].value_counts(dropna=False).items()
    }


def _missing_protocol_key_values(df: pd.DataFrame) -> list[str]:
    missing: list[str] = []
    for col in CELL_KEY_COLUMNS:
        text = df[col].fillna("").astype(str).str.strip()
        n_missing = int(text.eq("").sum())
        if n_missing:
            missing.append(f"{col}={n_missing}")
    return missing


def _has_columns(df: pd.DataFrame, columns: Sequence[str]) -> bool:
    return set(columns).issubset(df.columns)


def _valid_utc_timestamp_series(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip()
    parsed = pd.to_datetime(text, errors="coerce", utc=True)
    return text.ne("") & parsed.notna()


def _preview_rows(
    df: pd.DataFrame,
    mask: pd.Series,
    columns: Sequence[str],
    *,
    limit: int = 30,
) -> pd.DataFrame:
    """Return a compact bad-row preview without assuming all columns exist."""
    existing_cols = [c for c in columns if c in df.columns]
    if not existing_cols:
        return pd.DataFrame({"row_index": df.index[mask].tolist()[:limit]})
    return df.loc[mask, existing_cols].drop_duplicates().head(limit)


def audit_results(
    results_csv: Path,
    *,
    expected_cases: Sequence[str] | None = None,
    expected_methods: Sequence[str] | None = None,
    expected_budgets: Sequence[int] | None = None,
    expected_repeats: int = 30,
    expected_n_test: int | None = None,
    expected_test_seed: int | None = None,
    expected_method_families: dict[str, str] | None = None,
    seed_strategy: str | None = None,
    seed_start: int = 1000,
    min_repeats_per_cell: int = 28,
    min_ok_fraction: float = 0.95,
    max_timeout_fraction: float = 0.05,
    run_config_json: Path | None = None,
    require_run_config: bool = False,
) -> tuple[int, str]:
    """Return ``(exit_code, markdown_report)`` for one sweep CSV."""
    lines: list[str] = [
        "# Sweep Integrity Report",
        "",
        f"**Input**: `{results_csv}`",
        "",
    ]
    errors: list[str] = []
    warnings: list[str] = []
    run_config_notes: list[str] = []
    option_errors = _audit_option_errors(
        expected_budgets=expected_budgets,
        expected_repeats=expected_repeats,
        min_repeats_per_cell=min_repeats_per_cell,
        min_ok_fraction=min_ok_fraction,
        max_timeout_fraction=max_timeout_fraction,
    )
    if option_errors:
        errors.extend(option_errors)
        lines.extend(_render_findings(errors, warnings))
        return 1, "\n".join(lines) + "\n"

    if results_csv.is_symlink():
        errors.append(f"results.csv is a symlink and cannot be trusted: {results_csv}")
        lines.extend(_render_findings(errors, warnings))
        return 1, "\n".join(lines) + "\n"

    if not results_csv.exists():
        errors.append(f"results.csv does not exist: {results_csv}")
        lines.extend(_render_findings(errors, warnings))
        return 1, "\n".join(lines) + "\n"

    header_errors = csv_header_contract_errors(results_csv)
    if header_errors:
        errors.extend(header_errors)
        lines.extend(_render_findings(errors, warnings))
        return 1, "\n".join(lines) + "\n"

    try:
        df = pd.read_csv(results_csv, low_memory=False)
    except Exception as exc:  # pragma: no cover - defensive CLI path
        errors.append(f"could not parse results.csv: {type(exc).__name__}: {exc}")
        lines.extend(_render_findings(errors, warnings))
        return 1, "\n".join(lines) + "\n"

    if run_config_json is None:
        candidate = results_csv.with_name("run_config.json")
        run_config_json = candidate if candidate.exists() else None
    if run_config_json is not None:
        try:
            from scripts.check_sweep_runtime_health import _run_config_errors

            config_errors, config_warnings = _run_config_errors(df, run_config_json)
            errors.extend(config_errors)
            run_config_notes.extend(config_warnings)
        except Exception as exc:
            errors.append(
                "could not validate run_config.json startup snapshot: "
                f"{type(exc).__name__}: {exc}"
            )
    elif require_run_config:
        errors.append(
            "run_config.json is required for the final sweep integrity gate but "
            f"was not found next to results.csv: {results_csv.with_name('run_config.json')}"
        )

    missing_cols = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_cols:
        errors.append(f"missing required columns: {missing_cols}")

    n_rows = int(len(df))
    if n_rows == 0:
        errors.append(
            "results.csv has no data rows; wait for the first streamed cell, "
            "resume the sweep, or restore a non-empty server archive"
        )
    status_counts = _status_counts(df)
    ok_fraction = (
        float((df["status"] == "ok").mean())
        if "status" in df.columns and n_rows
        else 0.0
    )
    timeout_fraction = (
        float((df["status"] == "timeout").mean())
        if "status" in df.columns and n_rows
        else 0.0
    )

    observed_cases = sorted(df["case_id"].dropna().astype(str).unique()) if "case_id" in df else []
    observed_methods = sorted(df["method"].dropna().astype(str).unique()) if "method" in df else []
    budget_int, budget_is_integer = (
        strict_integer_series(df["budget_mult"])
        if "budget_mult" in df.columns
        else (
            pd.Series(pd.NA, index=df.index, dtype="Int64"),
            pd.Series(False, index=df.index),
        )
    )
    budget_numeric = pd.to_numeric(df["budget_mult"], errors="coerce") if "budget_mult" in df.columns else pd.Series(np.nan, index=df.index, dtype=float)
    observed_budgets = sorted(int(x) for x in budget_int.dropna().unique())

    lines.extend([
        "## Summary",
        "",
        f"- Rows: {n_rows}",
        f"- Cases observed: {len(observed_cases)}",
        f"- Methods observed: {len(observed_methods)}",
        f"- Budgets observed: {observed_budgets}",
        f"- Status counts: `{status_counts}`",
        f"- Failure-category counts: `{_failure_category_counts(df)}`",
        f"- OK fraction: {ok_fraction:.3f}",
        f"- Timeout fraction: {timeout_fraction:.3f}",
        "",
    ])

    if expected_cases is not None:
        _compare_sets("case", expected_cases, observed_cases, errors, warnings)
    if expected_methods is not None:
        _compare_sets("method", expected_methods, observed_methods, errors, warnings)
    if expected_budgets is not None:
        expected_budget_list = sorted(int(b) for b in expected_budgets)
        if observed_budgets != expected_budget_list:
            errors.append(
                f"budget set mismatch: expected {expected_budget_list}, observed {observed_budgets}"
            )

    if n_rows and _has_columns(df, CELL_KEY_COLUMNS):
        missing_keys = _missing_protocol_key_values(df)
        if missing_keys:
            errors.append(
                "protocol key values must be non-empty for every row; "
                f"missing values: {missing_keys}"
            )

    if seed_strategy and "seed" not in df.columns:
        errors.append(
            "seed_strategy was declared but results.csv has no seed column; "
            "paired-repeat reproducibility cannot be audited"
        )

    if ok_fraction < min_ok_fraction:
        errors.append(
            f"OK fraction {ok_fraction:.3f} < required {min_ok_fraction:.3f}"
        )
    if timeout_fraction > max_timeout_fraction:
        errors.append(
            f"timeout fraction {timeout_fraction:.3f} > allowed {max_timeout_fraction:.3f}"
        )

    if n_rows and _has_columns(df, ["case_id", "case_source_path", "case_source_sha256"]):
        case_source_mismatches: list[dict[str, str]] = []
        for case_id in observed_cases:
            expected_path, expected_sha, err = _current_case_source(case_id)
            if err is not None:
                errors.append(
                    f"current source hash could not be computed for case {case_id}: {err}"
                )
                continue
            mask = df["case_id"].astype(str) == case_id
            observed_paths = sorted(
                _normalize_source_path(x)
                for x in df.loc[mask, "case_source_path"].dropna().unique()
                if str(x).strip()
            )
            observed_hashes = sorted(
                str(x)
                for x in df.loc[mask, "case_source_sha256"].dropna().unique()
                if str(x).strip()
            )
            path_mismatch = not _source_paths_match(observed_paths, expected_path)
            hash_mismatch = observed_hashes != [expected_sha]
            if path_mismatch or hash_mismatch:
                path_preview = observed_paths[:5]
                hash_preview = observed_hashes[:5]
                if path_mismatch:
                    errors.append(
                        f"case_source_path mismatch for {case_id}: "
                        f"expected {expected_path}, observed {path_preview}"
                    )
                if hash_mismatch:
                    errors.append(
                        f"case_source_sha256 mismatch for {case_id}: "
                        f"expected current hash {expected_sha}, observed {hash_preview}"
                    )
                errors.append(
                    f"case source fingerprint mismatch for {case_id}; "
                    "results rows must match the current registered source path and hash"
                )
                case_source_mismatches.append({
                    "case_id": case_id,
                    "expected_path": str(expected_path),
                    "observed_path": ";".join(path_preview) or "<blank>",
                    "expected_sha256": str(expected_sha),
                    "observed_sha256": ";".join(hash_preview) or "<blank>",
                })
        if case_source_mismatches:
            lines.extend(
                _render_case_source_mismatches(
                    pd.DataFrame(case_source_mismatches)
                )
            )

    if n_rows and _has_columns(df, ["method", "method_source_path", "method_source_sha256"]):
        method_source_mismatches: list[dict[str, str]] = []
        current_methods = _current_method_sources(expected_methods or observed_methods)
        for method_name in observed_methods:
            current = current_methods.get(method_name)
            if current is None:
                errors.append(
                    f"current method source fingerprint could not be computed for {method_name}"
                )
                continue
            mask = df["method"].astype(str) == method_name
            observed_paths = sorted(
                _normalize_source_path(x)
                for x in df.loc[mask, "method_source_path"].dropna().unique()
                if str(x).strip()
            )
            observed_hashes = sorted(
                str(x)
                for x in df.loc[mask, "method_source_sha256"].dropna().unique()
                if str(x).strip()
            )
            expected_path = current["method_source_path"]
            expected_sha = current["method_source_sha256"]
            path_mismatch = not _source_paths_match(observed_paths, expected_path)
            hash_mismatch = observed_hashes != [expected_sha]
            if path_mismatch or hash_mismatch:
                path_preview = observed_paths[:5]
                hash_preview = observed_hashes[:5]
                if path_mismatch:
                    errors.append(
                        f"method_source_path mismatch for {method_name}: "
                        f"expected {expected_path}, observed {path_preview}"
                    )
                if hash_mismatch:
                    errors.append(
                        f"method_source_sha256 mismatch for {method_name}: "
                        f"expected current hash {expected_sha}, observed {hash_preview}"
                    )
                method_source_mismatches.append({
                    "method": method_name,
                    "expected_path": expected_path,
                    "observed_path": ";".join(path_preview) or "<blank>",
                    "expected_sha256": expected_sha,
                    "observed_sha256": ";".join(hash_preview) or "<blank>",
                })
        if method_source_mismatches:
            lines.extend(
                _render_method_source_mismatches(
                    pd.DataFrame(method_source_mismatches)
                )
            )

    if n_rows:
        expected_grid_cases = list(expected_cases) if expected_cases is not None else observed_cases
        expected_grid_methods = (
            list(expected_methods) if expected_methods is not None else observed_methods
        )
        expected_grid_budgets = (
            list(expected_budgets) if expected_budgets is not None else observed_budgets
        )

        repeat_int, repeat_is_integer = (
            strict_integer_series(df["repeat"])
            if "repeat" in df.columns
            else (
                pd.Series(pd.NA, index=df.index, dtype="Int64"),
                pd.Series(False, index=df.index),
            )
        )
        repeat_numeric = pd.to_numeric(df["repeat"], errors="coerce") if "repeat" in df.columns else pd.Series(np.nan, index=df.index, dtype=float)

    if n_rows and "status" in df.columns:
        bad_status_mask = ~df["status"].astype(str).isin(ALLOWED_STATUS)
        if bool(bad_status_mask.any()):
            bad_status = _preview_rows(
                df,
                bad_status_mask,
                CELL_KEY_COLUMNS + ["status"],
            )
            errors.append(
                f"status values must be one of {sorted(ALLOWED_STATUS)}; "
                f"found {int(bad_status_mask.sum())} invalid rows"
            )
            lines.extend(_render_invalid_statuses(bad_status))

    if n_rows and {"status", "error", "failure_category"}.issubset(df.columns):
        failure_category_text = df["failure_category"].fillna("").astype(str).str.strip()
        invalid_failure_category_mask = (
            failure_category_text.eq("")
            | ~failure_category_text.isin(ALLOWED_FAILURE_CATEGORIES)
        )
        if bool(invalid_failure_category_mask.any()):
            bad_categories = _preview_rows(
                df,
                invalid_failure_category_mask,
                CELL_KEY_COLUMNS + ["status", "failure_category", "error"],
            )
            errors.append(
                "failure_category values must be one of "
                f"{sorted(ALLOWED_FAILURE_CATEGORIES)}; found "
                f"{int(invalid_failure_category_mask.sum())} invalid rows"
            )
            lines.extend(
                _render_invalid_failure_categories(
                    bad_categories,
                    "Invalid Failure Categories",
                )
            )
        mismatch_mask = pd.Series(False, index=df.index)
        for idx in df.index:
            if not failure_category_is_consistent(
                df.at[idx, "status"],
                df.at[idx, "error"],
                df.at[idx, "failure_category"],
            ):
                mismatch_mask.loc[idx] = True
        if bool(mismatch_mask.any()):
            bad_contract = _preview_rows(
                df,
                mismatch_mask,
                CELL_KEY_COLUMNS + ["status", "failure_category", "error"],
            )
            errors.append(
                "failure_category must match the canonical status/error taxonomy; "
                f"found {int(mismatch_mask.sum())} inconsistent rows"
            )
            lines.extend(
                _render_invalid_failure_categories(
                    bad_contract,
                    "Failure Category Contract Mismatches",
                )
            )

    if n_rows:
        expected_cells = (
            len(expected_grid_cases) * len(expected_grid_methods) * len(expected_grid_budgets)
        )
        expected_rows = expected_cells * int(expected_repeats)
        if n_rows < expected_cells * int(min_repeats_per_cell):
            errors.append(
                "row count below minimum repeat floor: "
                f"{n_rows} < {expected_cells * int(min_repeats_per_cell)}"
            )
        if n_rows != expected_rows:
            errors.append(
                f"row count differs from declared full grid: expected {expected_rows}, observed {n_rows}"
            )

        expected_budget_set = {int(b) for b in expected_grid_budgets}
        budget_valid_shape = budget_is_integer
        budget_in_declared = pd.Series(False, index=df.index)
        budget_in_declared.loc[budget_valid_shape] = (
            budget_int.loc[budget_valid_shape].astype(int).isin(expected_budget_set)
        )
        bad_budget_mask = (
            ~budget_is_integer
            | (budget_valid_shape & ~budget_in_declared)
        )
        if bool(bad_budget_mask.any()):
            bad_budgets = _preview_rows(df, bad_budget_mask, CELL_KEY_COLUMNS)
            errors.append(
                f"budget_mult values must be integer members of {sorted(expected_budget_set)}; "
                f"found {int(bad_budget_mask.sum())} invalid rows"
            )
            lines.extend(_render_invalid_budgets(bad_budgets))

        df_for_cells = df.copy()
        df_for_cells["budget_mult"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
        df_for_cells.loc[budget_valid_shape, "budget_mult"] = (
            budget_int.loc[budget_valid_shape].astype(int).to_numpy()
        )
        if {"budget_mult", "dim", "n_train"}.issubset(df.columns):
            dim_int, dim_is_integer = strict_integer_series(df["dim"])
            n_train_int, n_train_is_integer = strict_integer_series(df["n_train"])
            valid_n_train_contract = (
                budget_valid_shape
                & dim_is_integer
                & (dim_int > 0)
                & n_train_is_integer
                & (n_train_int == budget_int * dim_int)
            )
            bad_n_train_mask = ~valid_n_train_contract
            if bool(bad_n_train_mask.any()):
                bad_n_train = _preview_rows(
                    df,
                    bad_n_train_mask,
                    CELL_KEY_COLUMNS + ["dim", "n_train"],
                )
                errors.append(
                    "n_train must equal budget_mult * dim with positive integer dim "
                    f"and integer n_train; found {int(bad_n_train_mask.sum())} invalid rows"
                )
                lines.extend(_render_invalid_n_train(bad_n_train))

        if "case_id" in df.columns:
            case_meta_errors = _case_metadata_mismatches(df)
            if not case_meta_errors.empty:
                errors.append(
                    "case metadata columns do not match the current registry; "
                    f"found {len(case_meta_errors)} mismatched case/column pairs"
                )
                lines.extend(_render_case_metadata_mismatches(case_meta_errors))

        if expected_method_families is not None:
            if "method" not in df.columns:
                errors.append(
                    "expected_method_families was declared but results.csv has no method column"
                )
            if "method_family" not in df.columns:
                errors.append(
                    "expected_method_families was declared but results.csv has no method_family column"
                )
            if {"method", "method_family"}.issubset(df.columns):
                fam_mismatches = _method_family_mismatches(df, expected_method_families)
                if not fam_mismatches.empty:
                    errors.append(
                        "method_family values do not match the declared method pool; "
                        f"found {len(fam_mismatches)} mismatched method/family pairs"
                    )
                    lines.extend(_render_method_family_mismatches(fam_mismatches))

        if expected_n_test is not None:
            if "n_test" not in df.columns:
                errors.append(
                    "expected_n_test was declared but results.csv has no n_test column"
                )
            else:
                n_test_int, n_test_is_integer = strict_integer_series(df["n_test"])
                bad_n_test_mask = (
                    ~n_test_is_integer
                    | (n_test_int != int(expected_n_test))
                )
                if bool(bad_n_test_mask.any()):
                    bad_n_test = _preview_rows(
                        df,
                        bad_n_test_mask,
                        CELL_KEY_COLUMNS + ["n_test"],
                    )
                    errors.append(
                        f"n_test must equal declared expected_n_test={int(expected_n_test)}; "
                        f"found {int(bad_n_test_mask.sum())} invalid rows"
                    )
                    lines.extend(_render_invalid_n_test(bad_n_test))

        if expected_test_seed is not None:
            if "test_seed" not in df.columns:
                errors.append(
                    "expected_test_seed was declared but results.csv has no test_seed column"
                )
            else:
                test_seed_int, test_seed_is_integer = strict_integer_series(df["test_seed"])
                bad_test_seed_mask = (
                    ~test_seed_is_integer
                    | (test_seed_int != int(expected_test_seed))
                )
                if bool(bad_test_seed_mask.any()):
                    bad_test_seed = _preview_rows(
                        df,
                        bad_test_seed_mask,
                        CELL_KEY_COLUMNS + ["test_seed"],
                    )
                    errors.append(
                        f"test_seed must equal declared expected_test_seed={int(expected_test_seed)}; "
                        f"found {int(bad_test_seed_mask.sum())} invalid rows"
                    )
                    lines.extend(_render_invalid_test_seed(bad_test_seed))

        if "repeat" in df.columns:
            bad_repeat_mask = (
                ~repeat_is_integer
                | (repeat_int < 0)
                | (repeat_int >= int(expected_repeats))
            )
            if bool(bad_repeat_mask.any()):
                bad_repeats = _preview_rows(df, bad_repeat_mask, CELL_KEY_COLUMNS)
                errors.append(
                    "repeat values must be integer IDs in "
                    f"[0, {int(expected_repeats) - 1}]; found {int(bad_repeat_mask.sum())} invalid rows"
                )
                lines.extend(_render_invalid_repeats(bad_repeats))

        if seed_strategy and _has_columns(df, ["case_id", "repeat", "seed"]):
            seed_int, seed_is_integer = strict_integer_series(df["seed"])
            valid_seed_shape = (
                seed_is_integer
                & repeat_is_integer
                & (repeat_int >= 0)
            )
            expected_seed = pd.Series(pd.NA, index=df.index, dtype="Int64")
            for idx in df.index[valid_seed_shape]:
                expected_seed.loc[idx] = _derive_seed(
                    str(df.at[idx, "case_id"]),
                    int(repeat_int.at[idx]),
                    strategy=seed_strategy,
                    seed_start=int(seed_start),
                )
            observed_seed = pd.Series(pd.NA, index=df.index, dtype="Int64")
            observed_seed.loc[valid_seed_shape] = (
                seed_int.loc[valid_seed_shape].astype(int).to_numpy()
            )
            bad_seed_mask = (
                ~seed_is_integer
                | (valid_seed_shape & observed_seed.ne(expected_seed))
            )
            if bool(bad_seed_mask.any()):
                bad_seeds = _preview_rows(
                    df,
                    bad_seed_mask,
                    CELL_KEY_COLUMNS + ["seed"],
                )
                errors.append(
                    f"seed values do not match {seed_strategy} seed derivation "
                    f"with seed_start={int(seed_start)}; found {int(bad_seed_mask.sum())} invalid rows"
                )
                lines.extend(_render_invalid_seeds(bad_seeds))

        protocol_cols = {
            "result_schema_version",
            "seed_strategy",
            "seed_start",
            "protocol_digest",
        }
        if _has_columns(df, protocol_cols):
            schema_version_int, schema_version_is_integer = strict_integer_series(
                df["result_schema_version"]
            )
            bad_schema_version_mask = (
                ~schema_version_is_integer
                | (schema_version_int != int(RESULT_SCHEMA_VERSION))
            )
            if bool(bad_schema_version_mask.any()):
                errors.append(
                    "result_schema_version must equal "
                    f"{int(RESULT_SCHEMA_VERSION)}; found "
                    f"{int(bad_schema_version_mask.sum())} invalid rows"
                )
            if seed_strategy:
                seed_strategy_text = (
                    df["seed_strategy"].fillna("").astype(str).str.strip()
                )
                bad_seed_strategy_mask = seed_strategy_text.ne(str(seed_strategy))
                if bool(bad_seed_strategy_mask.any()):
                    errors.append(
                        "seed_strategy values do not match declared "
                        f"{seed_strategy!r}; found "
                        f"{int(bad_seed_strategy_mask.sum())} invalid rows"
                    )
            seed_start_int, seed_start_is_integer = strict_integer_series(
                df["seed_start"]
            )
            bad_seed_start_mask = (
                ~seed_start_is_integer | (seed_start_int != int(seed_start))
            )
            if bool(bad_seed_start_mask.any()):
                errors.append(
                    "seed_start values do not match declared "
                    f"{int(seed_start)}; found "
                    f"{int(bad_seed_start_mask.sum())} invalid rows"
                )
            bad_digest_mask = invalid_protocol_digest_mask(df)
            if bool(bad_digest_mask.any()):
                bad_digests = _preview_rows(
                    df,
                    bad_digest_mask,
                    CELL_KEY_COLUMNS + ["protocol_digest"],
                )
                errors.append(
                    "protocol_digest must match the canonical row-level "
                    f"protocol payload; found {int(bad_digest_mask.sum())} "
                    "invalid rows"
                )
                lines.extend([
                    "",
                    "## Protocol Digest Mismatches",
                    "",
                    "```text",
                    str(bad_digests),
                    "```",
                    "",
                ])

        if _has_columns(df, CELL_KEY_COLUMNS):
            duplicate_mask = df.duplicated(CELL_KEY_COLUMNS, keep=False)
            if bool(duplicate_mask.any()):
                dup = (
                    df.loc[duplicate_mask, CELL_KEY_COLUMNS]
                    .value_counts()
                    .reset_index(name="n_rows")
                    .sort_values(CELL_KEY_COLUMNS)
                )
                errors.append(
                    f"{len(dup)} duplicate (case, method, budget, repeat) cells found"
                )
                lines.extend(_render_duplicate_cells(dup))

        if _has_columns(df_for_cells, CELL_KEY_COLUMNS):
            cell_counts = (
                df_for_cells.groupby(["case_id", "method", "budget_mult"])["repeat"]
                .nunique()
                .reset_index(name="n_repeats_observed")
            )
            low_repeat = cell_counts[cell_counts["n_repeats_observed"] < min_repeats_per_cell]
            missing_cell_rows = _find_missing_cells(
                cell_counts,
                expected_grid_cases,
                expected_grid_methods,
                expected_grid_budgets,
            )
            n_bad_cells = int(len(low_repeat) + len(missing_cell_rows))
            if n_bad_cells:
                errors.append(
                    f"{n_bad_cells} (case, method, budget) cells have fewer than "
                    f"{min_repeats_per_cell} repeats or are missing"
                )
                lines.extend(_render_bad_cells(low_repeat, missing_cell_rows))

        if "nrmse" in df.columns:
            ok = df["status"].eq("ok") if "status" in df.columns else pd.Series(True, index=df.index)
            non_ok = ~ok
            nrmse_numeric = pd.to_numeric(df["nrmse"], errors="coerce")
            bad_nrmse_mask = ok & (
                nrmse_numeric.isna()
                | ~np.isfinite(nrmse_numeric)
                | (nrmse_numeric < 0)
            )
            if bool(bad_nrmse_mask.any()):
                bad_nrmse = _preview_rows(
                    df,
                    bad_nrmse_mask,
                    CELL_KEY_COLUMNS + ["nrmse"],
                )
                errors.append(
                    "ok rows must have finite non-negative nrmse; "
                    f"found {int(bad_nrmse_mask.sum())} invalid rows"
                )
                lines.extend(_render_invalid_nrmse(bad_nrmse))
            nrmse_text = df["nrmse"].fillna("").astype(str).str.strip()
            non_ok_nrmse_mask = non_ok & nrmse_text.ne("")
            if bool(non_ok_nrmse_mask.any()):
                bad_non_ok_nrmse = _preview_rows(
                    df,
                    non_ok_nrmse_mask,
                    CELL_KEY_COLUMNS + ["status", "nrmse"],
                )
                errors.append(
                    "timeout/failed rows must leave nrmse blank so failure "
                    "penalties are applied only by the preregistered analysis; "
                    f"found {int(non_ok_nrmse_mask.sum())} invalid rows"
                )
                lines.extend(
                    _render_invalid_optional_metrics(
                        bad_non_ok_nrmse.assign(metric="nrmse")
                    )
                )

        ok = df["status"].eq("ok") if "status" in df.columns else pd.Series(True, index=df.index)
        non_ok = ~ok
        invalid_optional_metrics: list[pd.DataFrame] = []
        non_ok_optional_metrics: list[pd.DataFrame] = []
        for metric in OPTIONAL_OK_NUMERIC_METRICS:
            if metric not in df.columns:
                continue
            values = pd.to_numeric(df[metric], errors="coerce")
            bad_metric_mask = ok & (values.isna() | ~np.isfinite(values))
            if metric != "r2":
                bad_metric_mask = bad_metric_mask | (ok & (values < 0))
            if bool(bad_metric_mask.any()):
                preview = _preview_rows(
                    df,
                    bad_metric_mask,
                    CELL_KEY_COLUMNS + [metric],
                )
                preview.insert(0, "metric", metric)
                invalid_optional_metrics.append(preview)
                direction = "finite numeric" if metric == "r2" else "finite non-negative"
                errors.append(
                    f"ok rows with metric column {metric!r} must have {direction} values; "
                    f"found {int(bad_metric_mask.sum())} invalid rows"
                )
            metric_text = df[metric].fillna("").astype(str).str.strip()
            non_ok_metric_mask = non_ok & metric_text.ne("")
            if bool(non_ok_metric_mask.any()):
                preview = _preview_rows(
                    df,
                    non_ok_metric_mask,
                    CELL_KEY_COLUMNS + ["status", metric],
                )
                preview.insert(0, "metric", metric)
                non_ok_optional_metrics.append(preview)
                errors.append(
                    f"timeout/failed rows must leave metric column {metric!r} "
                    "blank so failure penalties are applied only by the "
                    "preregistered analysis; "
                    f"found {int(non_ok_metric_mask.sum())} invalid rows"
                )
        if invalid_optional_metrics:
            lines.extend(
                _render_invalid_optional_metrics(
                    pd.concat(invalid_optional_metrics, ignore_index=True)
                )
            )
        if non_ok_optional_metrics:
            lines.extend(
                _render_invalid_optional_metrics(
                    pd.concat(non_ok_optional_metrics, ignore_index=True)
                )
            )

        if "t_total_s" in df.columns:
            time_numeric = pd.to_numeric(df["t_total_s"], errors="coerce")
            bad_time_mask = (
                time_numeric.isna()
                | ~np.isfinite(time_numeric)
                | (time_numeric < 0)
            )
            if bool(bad_time_mask.any()):
                bad_times = _preview_rows(
                    df,
                    bad_time_mask,
                    CELL_KEY_COLUMNS + ["status", "t_total_s"],
                )
                errors.append(
                    "all rows must have finite non-negative t_total_s for "
                    f"compute-cost analysis; found {int(bad_time_mask.sum())} invalid rows"
                )
                lines.extend(_render_invalid_times(bad_times))

        if _has_columns(df, RUNTIME_TRACE_COLUMNS):
            worker_host = df["worker_host"].fillna("").astype(str).str.strip()
            bad_host_mask = worker_host.eq("")
            if bool(bad_host_mask.any()):
                bad_hosts = _preview_rows(
                    df,
                    bad_host_mask,
                    CELL_KEY_COLUMNS + ["worker_host"],
                )
                errors.append(
                    "all rows must include a non-empty worker_host for "
                    f"runtime provenance; found {int(bad_host_mask.sum())} invalid rows"
                )
                lines.extend(_render_invalid_runtime_trace(bad_hosts, "Invalid Worker Hosts"))

            worker_pid_int, worker_pid_is_integer = strict_integer_series(df["worker_pid"])
            bad_pid_mask = (
                ~worker_pid_is_integer
                | (worker_pid_int <= 0)
            )
            if bool(bad_pid_mask.any()):
                bad_pids = _preview_rows(
                    df,
                    bad_pid_mask,
                    CELL_KEY_COLUMNS + ["worker_pid"],
                )
                errors.append(
                    "all rows must include a positive-integer worker_pid for "
                    f"runtime provenance; found {int(bad_pid_mask.sum())} invalid rows"
                )
                lines.extend(_render_invalid_runtime_trace(bad_pids, "Invalid Worker PIDs"))

            for col in ("cell_started_utc", "cell_finished_utc"):
                bad_ts_mask = ~_valid_utc_timestamp_series(df[col])
                if bool(bad_ts_mask.any()):
                    bad_ts = _preview_rows(df, bad_ts_mask, CELL_KEY_COLUMNS + [col])
                    errors.append(
                        f"all rows must include parseable UTC {col}; "
                        f"found {int(bad_ts_mask.sum())} invalid rows"
                    )
                    lines.extend(_render_invalid_runtime_trace(bad_ts, f"Invalid {col}"))

            parsed_started = pd.to_datetime(
                df["cell_started_utc"].fillna("").astype(str).str.strip(),
                errors="coerce",
                utc=True,
            )
            parsed_finished = pd.to_datetime(
                df["cell_finished_utc"].fillna("").astype(str).str.strip(),
                errors="coerce",
                utc=True,
            )
            bad_time_order = parsed_finished < parsed_started
            if bool(bad_time_order.any()):
                bad_order = _preview_rows(
                    df,
                    bad_time_order,
                    CELL_KEY_COLUMNS + ["cell_started_utc", "cell_finished_utc"],
                )
                errors.append(
                    "cell_finished_utc must not be earlier than cell_started_utc; "
                    f"found {int(bad_time_order.sum())} invalid rows"
                )
                lines.extend(_render_invalid_runtime_trace(bad_order, "Invalid Runtime Time Order"))

            if "t_total_s" in df.columns:
                bad_duration = invalid_runtime_duration_mask(
                    df["t_total_s"],
                    parsed_started,
                    parsed_finished,
                )
                if bool(bad_duration.any()):
                    bad_duration_rows = _preview_rows(
                        df,
                        bad_duration,
                        CELL_KEY_COLUMNS
                        + ["t_total_s", "cell_started_utc", "cell_finished_utc"],
                    )
                    errors.append(
                        "t_total_s must agree with "
                        "cell_finished_utc - cell_started_utc within the runtime "
                        f"trace tolerance; found {int(bad_duration.sum())} invalid rows"
                    )
                    lines.extend(
                        _render_invalid_runtime_trace(
                            bad_duration_rows,
                            "Invalid Runtime Duration Consistency",
                        )
                    )

            rss_values: dict[str, pd.Series] = {}
            for col in ("rss_gb_start", "rss_gb_end", "rss_gb_peak"):
                values = pd.to_numeric(df[col], errors="coerce")
                rss_values[col] = values
                bad_rss_mask = values.isna() | ~np.isfinite(values) | (values < 0)
                if bool(bad_rss_mask.any()):
                    bad_rss = _preview_rows(df, bad_rss_mask, CELL_KEY_COLUMNS + [col])
                    errors.append(
                        f"all rows must include finite non-negative {col}; "
                        f"found {int(bad_rss_mask.sum())} invalid rows"
                    )
                    lines.extend(_render_invalid_runtime_trace(bad_rss, f"Invalid {col}"))
            bad_peak_order = (
                rss_values["rss_gb_peak"].notna()
                & np.isfinite(rss_values["rss_gb_peak"])
                & (
                    rss_values["rss_gb_peak"].lt(rss_values["rss_gb_start"])
                    | rss_values["rss_gb_peak"].lt(rss_values["rss_gb_end"])
                )
            )
            if bool(bad_peak_order.any()):
                bad_peak = _preview_rows(
                    df,
                    bad_peak_order,
                    CELL_KEY_COLUMNS + ["rss_gb_start", "rss_gb_end", "rss_gb_peak"],
                )
                errors.append(
                    "all rows must have rss_gb_peak >= rss_gb_start and "
                    f"rss_gb_end; found {int(bad_peak_order.sum())} invalid rows"
                )
                lines.extend(_render_invalid_runtime_trace(bad_peak, "Invalid RSS Peak Order"))

        if {"status", "error"}.issubset(df.columns):
            non_ok = ~df["status"].eq("ok")
            error_text = df["error"].fillna("").astype(str).str.strip()
            bad_error_mask = non_ok & error_text.eq("")
            if bool(bad_error_mask.any()):
                bad_errors = _preview_rows(
                    df,
                    bad_error_mask,
                    CELL_KEY_COLUMNS + ["status", "error"],
                )
                errors.append(
                    "timeout/failed rows must include a non-empty error reason; "
                    f"found {int(bad_error_mask.sum())} invalid rows"
                )
                lines.extend(_render_missing_errors(bad_errors))
            ok_with_error_mask = df["status"].eq("ok") & error_text.ne("")
            if bool(ok_with_error_mask.any()):
                ok_errors = _preview_rows(
                    df,
                    ok_with_error_mask,
                    CELL_KEY_COLUMNS + ["status", "error"],
                )
                errors.append(
                    "ok rows must leave error blank; "
                    f"found {int(ok_with_error_mask.sum())} invalid rows"
                )
                lines.extend(_render_ok_rows_with_errors(ok_errors))

    if run_config_notes:
        lines.extend(["", "## Run Config Binding", ""])
        lines.extend(f"- {note}" for note in run_config_notes)
        lines.append("")
    lines.extend(_render_findings(errors, warnings))
    rc = 1 if errors else 0
    return rc, "\n".join(lines) + "\n"


def _compare_sets(
    label: str,
    expected: Sequence[str],
    observed: Sequence[str],
    errors: list[str],
    warnings: list[str],
) -> None:
    expected_set = {str(x) for x in expected}
    observed_set = {str(x) for x in observed}
    missing = sorted(expected_set - observed_set)
    extra = sorted(observed_set - expected_set)
    if missing:
        errors.append(f"missing {label}s: {missing}")
    if extra:
        errors.append(f"unexpected extra {label}s: {extra}")


def _audit_option_errors(
    *,
    expected_budgets: Sequence[int] | None,
    expected_repeats: int,
    min_repeats_per_cell: int,
    min_ok_fraction: float,
    max_timeout_fraction: float,
) -> list[str]:
    errors: list[str] = []
    for name, value in (
        ("expected_repeats", expected_repeats),
        ("min_repeats_per_cell", min_repeats_per_cell),
    ):
        if isinstance(value, bool):
            errors.append(f"{name} must be an integer; got {value!r}")
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            errors.append(f"{name} must be an integer; got {value!r}")
            continue
        if name == "expected_repeats" and parsed < 1:
            errors.append(f"{name} must be >= 1; got {value!r}")
        if name == "min_repeats_per_cell" and parsed < 0:
            errors.append(f"{name} must be >= 0; got {value!r}")

    try:
        if int(min_repeats_per_cell) > int(expected_repeats):
            errors.append(
                "min_repeats_per_cell must not exceed expected_repeats; "
                f"got {min_repeats_per_cell!r} > {expected_repeats!r}"
            )
    except (TypeError, ValueError):
        pass

    for name, value in (
        ("min_ok_fraction", min_ok_fraction),
        ("max_timeout_fraction", max_timeout_fraction),
    ):
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            errors.append(f"{name} must be a finite fraction in [0, 1]; got {value!r}")
            continue
        if not np.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
            errors.append(f"{name} must be a finite fraction in [0, 1]; got {value!r}")

    if expected_budgets is not None:
        parsed_budgets: list[int] = []
        for raw in expected_budgets:
            if isinstance(raw, bool):
                errors.append(f"expected_budgets must contain positive integers; got {raw!r}")
                continue
            try:
                parsed = int(raw)
            except (TypeError, ValueError):
                errors.append(f"expected_budgets must contain positive integers; got {raw!r}")
                continue
            if parsed <= 0:
                errors.append(f"expected_budgets must contain positive integers; got {raw!r}")
            parsed_budgets.append(parsed)
        if parsed_budgets and len(set(parsed_budgets)) != len(parsed_budgets):
            errors.append(f"expected_budgets must not contain duplicates; got {parsed_budgets}")
    return errors


def _find_missing_cells(
    cell_counts: pd.DataFrame,
    cases: Sequence[str],
    methods: Sequence[str],
    budgets: Sequence[int],
) -> pd.DataFrame:
    seen = {
        (str(r.case_id), str(r.method), int(r.budget_mult))
        for r in cell_counts.itertuples(index=False)
    }
    rows = [
        {"case_id": c, "method": m, "budget_mult": int(b), "n_repeats_observed": 0}
        for c in cases
        for m in methods
        for b in budgets
        if (str(c), str(m), int(b)) not in seen
    ]
    return pd.DataFrame(rows)


def _render_bad_cells(low_repeat: pd.DataFrame, missing: pd.DataFrame) -> list[str]:
    combined = pd.concat([low_repeat, missing], ignore_index=True)
    if combined.empty:
        return []
    preview = combined.sort_values(["case_id", "method", "budget_mult"]).head(30)
    lines = ["", "## Cells Below Repeat Floor", "", preview.to_markdown(index=False), ""]
    if len(combined) > len(preview):
        lines.append(f"... {len(combined) - len(preview)} more cells omitted from preview.")
        lines.append("")
    return lines


def _render_duplicate_cells(duplicates: pd.DataFrame) -> list[str]:
    if duplicates.empty:
        return []
    preview = duplicates.head(30)
    lines = ["", "## Duplicate Cells", "", preview.to_markdown(index=False), ""]
    if len(duplicates) > len(preview):
        lines.append(f"... {len(duplicates) - len(preview)} more duplicate cells omitted from preview.")
        lines.append("")
    return lines


def _render_invalid_budgets(invalid: pd.DataFrame) -> list[str]:
    if invalid.empty:
        return []
    return ["", "## Invalid Budgets", "", invalid.to_markdown(index=False), ""]


def _render_invalid_statuses(invalid: pd.DataFrame) -> list[str]:
    if invalid.empty:
        return []
    return ["", "## Invalid Status Values", "", invalid.to_markdown(index=False), ""]


def _render_invalid_failure_categories(invalid: pd.DataFrame, title: str) -> list[str]:
    if invalid.empty:
        return []
    return ["", f"## {title}", "", invalid.to_markdown(index=False), ""]


def _render_invalid_n_train(invalid: pd.DataFrame) -> list[str]:
    if invalid.empty:
        return []
    return ["", "## Invalid Training Sizes", "", invalid.to_markdown(index=False), ""]


def _case_metadata_mismatches(df: pd.DataFrame) -> pd.DataFrame:
    checks = {
        "domain": lambda meta: meta.domain,
        "dim": lambda meta: int(meta.dim_input),
        "n_output": lambda meta: int(meta.dim_output),
    }
    rows: list[dict[str, object]] = []
    for case_id in sorted(df["case_id"].dropna().astype(str).unique()):
        try:
            meta = db.get_case_meta(case_id)
        except Exception:
            continue
        sub = df[df["case_id"].astype(str) == case_id]
        for col, expected_fn in checks.items():
            if col not in sub.columns:
                continue
            observed_values = {
                str(v)
                for v in sub[col].dropna().unique()
                if str(v).strip() != ""
            }
            expected = expected_fn(meta)
            expected_text = str(expected)
            if col in {"dim", "n_output"}:
                numeric = pd.to_numeric(sub[col], errors="coerce")
                mismatch = bool(numeric.isna().any()) or bool(
                    numeric.dropna().astype(float).ne(float(expected)).any()
                )
            else:
                mismatch = observed_values != {expected_text}
            if mismatch:
                rows.append({
                    "case_id": case_id,
                    "column": col,
                    "expected": expected_text,
                    "observed": ";".join(sorted(observed_values)) or "<blank>",
                })
    return pd.DataFrame(rows)


def _render_case_metadata_mismatches(invalid: pd.DataFrame) -> list[str]:
    if invalid.empty:
        return []
    preview = invalid.head(30)
    lines = ["", "## Case Metadata Mismatches", "", preview.to_markdown(index=False), ""]
    if len(invalid) > len(preview):
        lines.append(f"... {len(invalid) - len(preview)} more mismatches omitted from preview.")
        lines.append("")
    return lines


def _render_case_source_mismatches(invalid: pd.DataFrame) -> list[str]:
    if invalid.empty:
        return []
    preview = invalid.head(30)
    lines = ["", "## Case Source Fingerprint Mismatches", "", preview.to_markdown(index=False), ""]
    if len(invalid) > len(preview):
        lines.append(f"... {len(invalid) - len(preview)} more mismatches omitted from preview.")
        lines.append("")
    return lines


def _render_method_source_mismatches(invalid: pd.DataFrame) -> list[str]:
    if invalid.empty:
        return []
    preview = invalid.head(30)
    lines = ["", "## Method Source Fingerprint Mismatches", "", preview.to_markdown(index=False), ""]
    if len(invalid) > len(preview):
        lines.append(f"... {len(invalid) - len(preview)} more mismatches omitted from preview.")
        lines.append("")
    return lines


def _method_family_mismatches(
    df: pd.DataFrame,
    expected: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for method in sorted(df["method"].dropna().astype(str).unique()):
        if method not in expected:
            continue
        sub = df[df["method"].astype(str) == method]
        observed = {
            str(v)
            for v in sub["method_family"].dropna().unique()
            if str(v).strip() != ""
        }
        if observed != {str(expected[method])}:
            rows.append({
                "method": method,
                "expected": str(expected[method]),
                "observed": ";".join(sorted(observed)) or "<blank>",
            })
    return pd.DataFrame(rows)


def _render_method_family_mismatches(invalid: pd.DataFrame) -> list[str]:
    if invalid.empty:
        return []
    return ["", "## Method Family Mismatches", "", invalid.head(30).to_markdown(index=False), ""]


def _render_invalid_n_test(invalid: pd.DataFrame) -> list[str]:
    if invalid.empty:
        return []
    return ["", "## Invalid Test Set Sizes", "", invalid.to_markdown(index=False), ""]


def _render_invalid_test_seed(invalid: pd.DataFrame) -> list[str]:
    if invalid.empty:
        return []
    return ["", "## Invalid Test Seeds", "", invalid.to_markdown(index=False), ""]


def _render_invalid_repeats(invalid: pd.DataFrame) -> list[str]:
    if invalid.empty:
        return []
    return ["", "## Invalid Repeat IDs", "", invalid.to_markdown(index=False), ""]


def _render_invalid_seeds(invalid: pd.DataFrame) -> list[str]:
    if invalid.empty:
        return []
    return ["", "## Invalid Seeds", "", invalid.to_markdown(index=False), ""]


def _render_invalid_nrmse(invalid: pd.DataFrame) -> list[str]:
    if invalid.empty:
        return []
    return ["", "## Invalid NRMSE Values", "", invalid.to_markdown(index=False), ""]


def _render_invalid_optional_metrics(invalid: pd.DataFrame) -> list[str]:
    if invalid.empty:
        return []
    return ["", "## Invalid Optional Metric Values", "", invalid.head(30).to_markdown(index=False), ""]


def _render_invalid_times(invalid: pd.DataFrame) -> list[str]:
    if invalid.empty:
        return []
    return ["", "## Invalid Timing Values", "", invalid.to_markdown(index=False), ""]


def _render_invalid_runtime_trace(invalid: pd.DataFrame, title: str) -> list[str]:
    if invalid.empty:
        return []
    return ["", f"## {title}", "", invalid.to_markdown(index=False), ""]


def _render_missing_errors(invalid: pd.DataFrame) -> list[str]:
    if invalid.empty:
        return []
    return ["", "## Missing Failure Reasons", "", invalid.to_markdown(index=False), ""]


def _render_ok_rows_with_errors(invalid: pd.DataFrame) -> list[str]:
    if invalid.empty:
        return []
    return ["", "## OK Rows With Error Text", "", invalid.to_markdown(index=False), ""]


def _derive_seed(case_id: str, repeat: int, *, strategy: str, seed_start: int) -> int:
    if strategy == "legacy":
        return int(seed_start + repeat)
    if strategy != "case_repeat":
        raise ValueError(f"unknown seed strategy {strategy!r}")
    payload = f"{seed_start}|{case_id}|{repeat}".encode()
    digest = hashlib.blake2b(payload, digest_size=4).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) & 0x7FFFFFFF


def _render_findings(errors: Iterable[str], warnings: Iterable[str]) -> list[str]:
    errors = list(errors)
    warnings = list(warnings)
    lines = ["", "## Findings", ""]
    if not errors and not warnings:
        lines.append("PASS: sweep satisfies the declared integrity gate.")
        return lines
    if errors:
        lines.append("### Errors")
        lines.append("")
        lines.extend(f"- {e}" for e in errors)
        lines.append("")
    if warnings:
        lines.append("### Warnings")
        lines.append("")
        lines.extend(f"- {w}" for w in warnings)
        lines.append("")
    return lines


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit DOE-Benchmark results.csv completeness before paper stages.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("results_csv", type=Path)
    p.add_argument("--expected-cases", default=None, help="Comma-separated case IDs.")
    p.add_argument("--expected-methods", default=None, help="Comma-separated method names.")
    p.add_argument(
        "--expected-method-families",
        default=None,
        help="Comma-separated method:family pairs used to validate method_family.",
    )
    p.add_argument("--expected-budgets", type=int, nargs="+", default=[5, 10, 20, 50])
    p.add_argument("--expected-repeats", type=int, default=30)
    p.add_argument(
        "--expected-n-test",
        type=int,
        default=None,
        help="Require per-row n_test to match this declared test-set size.",
    )
    p.add_argument(
        "--expected-test-seed",
        type=int,
        default=None,
        help="Require per-row test_seed to match this declared reference-set seed.",
    )
    p.add_argument("--seed-strategy", choices=["case_repeat", "legacy"], default=None)
    p.add_argument("--seed-start", type=int, default=1000)
    p.add_argument("--min-repeats-per-cell", type=int, default=28)
    p.add_argument("--min-ok-fraction", type=float, default=0.95)
    p.add_argument("--max-timeout-fraction", type=float, default=0.05)
    p.add_argument(
        "--run-config-json",
        type=Path,
        default=None,
        help=(
            "Stage-2 run_config.json startup snapshot to bind the final "
            "results.csv archive to. Defaults to a sibling run_config.json "
            "when present."
        ),
    )
    p.add_argument(
        "--require-run-config",
        action="store_true",
        help=(
            "Fail unless a Stage-2 run_config.json startup snapshot is present "
            "and matches results.csv. Formal pipeline Stage 2b should use this."
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional Markdown report path. Defaults to stdout only.",
    )
    return p.parse_args(argv)


def write_report_atomic(report: str, output: Path) -> None:
    """Replace the audit report only after the new report is fully written."""
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp.", dir=str(output.parent))
    )
    tmp_output = tmp_dir / output.name
    try:
        tmp_output.write_text(report, encoding="utf-8")
        shutil.move(str(tmp_output), str(output))
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    expected_cases = _split_csv_arg(args.expected_cases)
    expected_methods = _split_csv_arg(args.expected_methods)
    expected_method_families = _split_mapping_arg(args.expected_method_families)
    rc, report = audit_results(
        args.results_csv,
        expected_cases=expected_cases,
        expected_methods=expected_methods,
        expected_budgets=args.expected_budgets,
        expected_repeats=args.expected_repeats,
        expected_n_test=args.expected_n_test,
        expected_test_seed=args.expected_test_seed,
        expected_method_families=expected_method_families,
        seed_strategy=args.seed_strategy,
        seed_start=args.seed_start,
        min_repeats_per_cell=args.min_repeats_per_cell,
        min_ok_fraction=args.min_ok_fraction,
        max_timeout_fraction=args.max_timeout_fraction,
        run_config_json=args.run_config_json,
        require_run_config=bool(args.require_run_config),
    )
    if args.output is not None:
        write_report_atomic(report, args.output)
    print(report)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
