"""
scripts/prepare_figdata.py
===========================
从 results.csv + D1/D2/D3 分析输出直接生成所有论文图表所需数据 CSV。
注意：不依赖 generate_report.py 的输出，直接调用 benchmark 库函数。

Usage::

    python scripts/prepare_figdata.py \
        --results-csv   benchmark/results/full_benchmark/results.csv \
        --d1-dir        benchmark/results/ranking_reversal \
        --d2-dir        benchmark/results/sample_complexity_0.20 \
        --d3-dir        benchmark/results/reliability_recovery \
        --out-dir       ../manuscripts/figdata
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.project_layout import FIGDATA_DIR, portable_path  # noqa: E402

import numpy as np
import pandas as pd
from benchmark.runtime_contracts import (
    ALLOWED_FAILURE_CATEGORIES,
    PROTOCOL_DIGEST_COLUMNS,
    RESULT_SCHEMA_VERSION,
    failure_category_is_consistent,
    invalid_protocol_digest_mask,
    invalid_runtime_duration_mask,
)

D1_ANALYSIS = "complete_case"
D2_ANALYSIS = "penalized_failed_in_denominator"
PRIMARY_FIG6_ANALYSIS = "penalized_failed_in_denominator"
D3_PRIMARY_CLUSTER_BY = "case_id"
D3_SENSITIVITY_CLUSTER_BY = "iid"
ANALYSIS_POLICY = {
    "schema_version": 1,
    "d1_primary_analysis": D1_ANALYSIS,
    "d2_primary_analysis": D2_ANALYSIS,
    "fig6_primary_analysis": PRIMARY_FIG6_ANALYSIS,
    "d3_primary_cluster_by": D3_PRIMARY_CLUSTER_BY,
    "d3_sensitivity_cluster_by": D3_SENSITIVITY_CLUSTER_BY,
    "failure_policy": "penalized_failed_in_denominator for D2 and Fig. 6 primary",
    "d1_failure_policy": "complete_case",
}
HEADLINE_RESULTS_CSV = ROOT / "benchmark" / "results" / "full_benchmark" / "results.csv"
HEADLINE_BUDGETS = {5, 10, 20, 50}
HEADLINE_REPEATS = set(range(30))
HEADLINE_N_METHODS = 20
REQUIRED_RESULTS_COLUMNS = {
    "case_id",
    "method",
    "budget_mult",
    "repeat",
    "status",
    "failure_category",
    "error",
    "nrmse",
} | PROTOCOL_DIGEST_COLUMNS
RUNTIME_TRACE_COLUMNS = {
    "worker_host",
    "worker_pid",
    "cell_started_utc",
    "cell_finished_utc",
    "rss_gb_start",
    "rss_gb_end",
    "rss_gb_peak",
}
ALLOWED_STATUS = {"ok", "timeout", "failed"}
TIMING_COLUMNS = [
    "t_total_s",
    "wall_time_s",
    "wallclock_s",
    "t_predict_s",
    "t_fit_s",
]


def _split_csv_arg(value: str | None) -> list[str]:
    if not value:
        return []
    return [x.strip() for x in value.replace(";", ",").split(",") if x.strip()]


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _save(df: pd.DataFrame, path: Path, source: str) -> None:
    df = df.copy()
    df["_source"] = source
    df["_generated_at"] = _ts()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  [ok] {path.name}: {len(df)} rows")


def _holm_adjust(pvals: np.ndarray) -> np.ndarray:
    """Step-down Holm-Bonferroni adjustment matching benchmark.reliability_recovery."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return out
    vals = p[finite]
    order = np.argsort(vals)
    m = len(vals)
    adjusted = np.empty_like(vals)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * vals[idx])
        adjusted[idx] = min(running, 1.0)
    out[finite] = adjusted
    return out


def validate_results_methods(
    df: pd.DataFrame,
    expected_methods: list[str] | tuple[str, ...] | set[str],
) -> None:
    """Reject result CSVs that contain methods outside the declared pool.

    Stage 8 is a derivative-artifact builder, so its local guard is scoped to
    contamination prevention.  Completeness of the full method grid is checked
    earlier by ``audit_sweep_integrity.py``.
    """
    if not expected_methods:
        return
    assert "method" in df.columns, "results.csv missing method column"
    expected = {str(m) for m in expected_methods}
    observed = set(df["method"].dropna().astype(str).unique())
    extra = sorted(observed - expected)
    if extra:
        raise AssertionError(
            "results.csv contains methods outside the declared pool: "
            f"{extra}. Regenerate or filter the sweep before building figdata."
        )


def current_headline_methods() -> list[str]:
    """Return the frozen headline ``nature_pool()`` method names."""
    from benchmark.methods import nature_pool

    methods = [str(m.name) for m in nature_pool()]
    if len(methods) != HEADLINE_N_METHODS:
        raise AssertionError(
            "headline paper artifacts require nature_pool() to expose exactly "
            f"{HEADLINE_N_METHODS} methods; found {len(methods)}"
        )
    return methods


def validate_headline_results_source(
    results_csv: Path,
    df: pd.DataFrame,
    *,
    declared_methods: list[str] | tuple[str, ...] | set[str] | None = None,
) -> None:
    """Validate that formal paper artifacts consume the canonical headline grid.

    This is intentionally stricter than ``validate_results_contract``.  The
    row-level contract catches malformed CSVs; this guard catches wrong-source
    or wrong-pool artifacts before SOTA/MF robustness rows can be turned into
    main-paper figdata or numbers.
    """
    if results_csv.resolve() != HEADLINE_RESULTS_CSV.resolve():
        raise AssertionError(
            "headline paper artifacts must read the canonical results source "
            f"{portable_path(HEADLINE_RESULTS_CSV, ROOT)}; got {portable_path(results_csv, ROOT)}"
        )

    expected_methods = set(current_headline_methods())
    if declared_methods:
        declared = {str(m) for m in declared_methods}
        if declared != expected_methods:
            raise AssertionError(
                "headline declared methods do not match the current 20-method "
                f"nature_pool(); missing={sorted(expected_methods - declared)}, "
                f"extra={sorted(declared - expected_methods)}"
            )

    import doe_testbed as dbt

    expected_cases = {str(c.case_id) for c in dbt.list_cases()}
    observed_cases = set(df["case_id"].dropna().astype(str).unique())
    observed_methods = set(df["method"].dropna().astype(str).unique())
    observed_budgets = set(_integer_series(df["budget_mult"], "budget_mult").unique())
    observed_repeats = set(_integer_series(df["repeat"], "repeat").unique())

    problems: list[str] = []
    if observed_cases != expected_cases:
        problems.append(
            f"case set drift: missing={sorted(expected_cases - observed_cases)}, "
            f"extra={sorted(observed_cases - expected_cases)}"
        )
    if observed_methods != expected_methods:
        problems.append(
            f"method set drift: missing={sorted(expected_methods - observed_methods)}, "
            f"extra={sorted(observed_methods - expected_methods)}"
        )
    if observed_budgets != HEADLINE_BUDGETS:
        problems.append(
            f"budget set drift: expected={sorted(HEADLINE_BUDGETS)}, "
            f"observed={sorted(observed_budgets)}"
        )
    if observed_repeats != HEADLINE_REPEATS:
        missing_repeats = sorted(HEADLINE_REPEATS - observed_repeats)
        extra_repeats = sorted(observed_repeats - HEADLINE_REPEATS)
        problems.append(
            f"repeat set drift: missing={missing_repeats}, extra={extra_repeats}"
        )

    expected_rows = (
        len(expected_cases)
        * len(expected_methods)
        * len(HEADLINE_BUDGETS)
        * len(HEADLINE_REPEATS)
    )
    if len(df) != expected_rows:
        problems.append(
            f"row-count drift: expected {expected_rows}, observed {len(df)}"
        )

    if problems:
        raise AssertionError(
            "headline results grid is not the frozen "
            "34-case x 20-method x 4-budget x 30-repeat archive: "
            + "; ".join(problems)
        )


def validate_results_contract(df: pd.DataFrame) -> None:
    """Fail fast on results.csv corruption before any figdata is written.

    The full sweep audit remains the authoritative grid-completeness gate.
    This local Stage 8 check protects downstream paper artifacts from malformed
    or partially hand-edited CSVs when figdata is run directly.
    """
    missing = sorted(REQUIRED_RESULTS_COLUMNS - set(df.columns))
    if missing:
        raise AssertionError(f"results.csv missing required columns: {missing}")

    if df.empty:
        raise AssertionError("results.csv is empty")

    missing_keys = _missing_protocol_key_values(df)
    if missing_keys:
        raise AssertionError(
            "results.csv protocol key values must be non-empty; "
            f"missing values: {missing_keys}"
        )

    schema_version = _integer_series(
        df["result_schema_version"], "result_schema_version"
    )
    bad_schema = schema_version.ne(int(RESULT_SCHEMA_VERSION))
    if bool(bad_schema.any()):
        examples = _cell_examples(df.loc[bad_schema])
        raise AssertionError(
            "results.csv result_schema_version must equal "
            f"{int(RESULT_SCHEMA_VERSION)}; found "
            f"{int(bad_schema.sum())} invalid row(s): {examples}"
        )
    seed_strategy = df["seed_strategy"].fillna("").astype(str).str.strip()
    missing_seed_strategy = seed_strategy.eq("")
    if bool(missing_seed_strategy.any()):
        examples = _cell_examples(df.loc[missing_seed_strategy])
        raise AssertionError(
            "results.csv seed_strategy must be non-empty; "
            f"found {int(missing_seed_strategy.sum())} invalid row(s): {examples}"
        )
    _integer_series(df["seed_start"], "seed_start")
    bad_digest = invalid_protocol_digest_mask(df)
    if bool(bad_digest.any()):
        examples = _cell_examples(df.loc[bad_digest])
        raise AssertionError(
            "results.csv protocol_digest must match the canonical row-level "
            f"protocol payload; found {int(bad_digest.sum())} invalid row(s): "
            f"{examples}"
        )

    budget_mult = _integer_series(df["budget_mult"], "budget_mult")
    repeat = _integer_series(df["repeat"], "repeat")
    duplicate_cells = pd.DataFrame({
        "case_id": df["case_id"].astype(str),
        "method": df["method"].astype(str),
        "budget_mult": budget_mult.astype("Int64"),
        "repeat": repeat.astype("Int64"),
    }).duplicated(["case_id", "method", "budget_mult", "repeat"], keep=False)
    if bool(duplicate_cells.any()):
        examples = _cell_examples(df.loc[duplicate_cells])
        raise AssertionError(
            "results.csv contains duplicate "
            "(case_id, method, budget_mult, repeat) cells: "
            f"{examples}"
        )

    bad_status = sorted(
        set(df["status"].dropna().astype(str).unique()) - ALLOWED_STATUS
    )
    if bad_status:
        raise AssertionError(
            "results.csv status values must be one of "
            f"{sorted(ALLOWED_STATUS)}; found {bad_status}"
        )
    if df["status"].isna().any():
        raise AssertionError("results.csv status column contains missing values")

    ok = df["status"].astype(str).eq("ok")
    if not bool(ok.any()):
        raise AssertionError("results.csv contains no status='ok' rows")

    nrmse = pd.to_numeric(df["nrmse"], errors="coerce")
    bad_ok_nrmse = ok & (nrmse.isna() | ~np.isfinite(nrmse) | nrmse.lt(0))
    if bool(bad_ok_nrmse.any()):
        examples = _cell_examples(df.loc[bad_ok_nrmse])
        raise AssertionError(
            "ok rows must have finite non-negative nrmse; "
            f"found {int(bad_ok_nrmse.sum())} invalid row(s): {examples}"
        )
    non_ok = ~ok
    nrmse_text = df["nrmse"].fillna("").astype(str).str.strip()
    non_ok_nrmse = non_ok & nrmse_text.ne("")
    if bool(non_ok_nrmse.any()):
        examples = _cell_examples(df.loc[non_ok_nrmse])
        raise AssertionError(
            "timeout/failed rows must leave nrmse blank so failure penalties "
            "are applied only by the preregistered analysis; "
            f"found {int(non_ok_nrmse.sum())} invalid row(s): {examples}"
        )

    for metric in ("rmse", "mae", "maxae", "q95ae", "r2"):
        if metric not in df.columns:
            continue
        metric_text = df[metric].fillna("").astype(str).str.strip()
        non_ok_metric = non_ok & metric_text.ne("")
        if bool(non_ok_metric.any()):
            examples = _cell_examples(df.loc[non_ok_metric])
            raise AssertionError(
                f"timeout/failed rows must leave {metric} blank so failure "
                "penalties are applied only by the preregistered analysis; "
                f"found {int(non_ok_metric.sum())} invalid row(s): {examples}"
            )

    error_text = df["error"].fillna("").astype(str).str.strip()
    ok_with_error = ok & error_text.ne("")
    if bool(ok_with_error.any()):
        examples = _cell_examples(df.loc[ok_with_error])
        raise AssertionError(
            "ok rows must have blank error; "
            f"found {int(ok_with_error.sum())} invalid row(s): {examples}"
        )
    failed_without_error = df["status"].astype(str).isin({"timeout", "failed"}) & error_text.eq("")
    if bool(failed_without_error.any()):
        examples = _cell_examples(df.loc[failed_without_error])
        raise AssertionError(
            "timeout/failed rows must include a non-empty error reason; "
            f"found {int(failed_without_error.sum())} invalid row(s): {examples}"
        )
    failure_category = df["failure_category"].fillna("").astype(str).str.strip()
    invalid_failure_category = failure_category.eq("") | ~failure_category.isin(
        ALLOWED_FAILURE_CATEGORIES
    )
    if bool(invalid_failure_category.any()):
        examples = _cell_examples(df.loc[invalid_failure_category])
        raise AssertionError(
            "failure_category values must be one of "
            f"{sorted(ALLOWED_FAILURE_CATEGORIES)}; found "
            f"{int(invalid_failure_category.sum())} invalid row(s): {examples}"
        )
    failure_category_mismatch = pd.Series(False, index=df.index)
    for idx in df.index:
        if not failure_category_is_consistent(
            df.at[idx, "status"],
            df.at[idx, "error"],
            df.at[idx, "failure_category"],
        ):
            failure_category_mismatch.loc[idx] = True
    if bool(failure_category_mismatch.any()):
        examples = _cell_examples(df.loc[failure_category_mismatch])
        raise AssertionError(
            "failure_category must match the canonical status/error taxonomy; "
            f"found {int(failure_category_mismatch.sum())} invalid row(s): {examples}"
        )

    cost_col = next((col for col in TIMING_COLUMNS if col in df.columns), None)
    if cost_col is None:
        raise AssertionError("results.csv has no timing column for paper figdata")
    timing = pd.to_numeric(df[cost_col], errors="coerce")
    bad_timing = timing.isna() | ~np.isfinite(timing) | timing.lt(0)
    if bool(bad_timing.any()):
        examples = _cell_examples(df.loc[bad_timing])
        raise AssertionError(
            f"all rows must have finite non-negative {cost_col}; "
            f"found {int(bad_timing.sum())} invalid row(s): {examples}"
        )
    bad_ok_timing = ok & (timing.isna() | ~np.isfinite(timing) | timing.le(0))
    if bool(bad_ok_timing.any()):
        examples = _cell_examples(df.loc[bad_ok_timing])
        raise AssertionError(
            f"ok rows must have finite positive {cost_col}; "
            f"found {int(bad_ok_timing.sum())} invalid row(s): {examples}"
        )

    _validate_runtime_trace(df)


def _integer_series(df_col: pd.Series, label: str) -> pd.Series:
    numeric = pd.to_numeric(df_col, errors="coerce")
    valid = numeric.notna() & np.isfinite(numeric) & numeric.eq(np.floor(numeric))
    if not bool(valid.all()):
        bad = df_col[~valid].head(10).astype(str).tolist()
        raise AssertionError(
            f"results.csv {label} values must be finite integers; "
            f"invalid values: {bad}"
        )
    return numeric.astype(int)


def _validate_runtime_trace(df: pd.DataFrame) -> None:
    """Require per-cell runtime provenance before paper artifacts are derived."""
    missing = sorted(RUNTIME_TRACE_COLUMNS - set(df.columns))
    if missing:
        raise AssertionError(
            "results.csv missing runtime provenance columns: "
            f"{missing}"
        )

    worker_host = df["worker_host"].fillna("").astype(str).str.strip()
    bad_host = worker_host.eq("")
    if bool(bad_host.any()):
        examples = _cell_examples(df.loc[bad_host])
        raise AssertionError(
            "all rows must include a non-empty worker_host; "
            f"found {int(bad_host.sum())} invalid row(s): {examples}"
        )

    worker_pid = pd.to_numeric(df["worker_pid"], errors="coerce")
    bad_pid = (
        worker_pid.isna()
        | ~np.isfinite(worker_pid)
        | worker_pid.ne(np.floor(worker_pid))
        | worker_pid.le(0)
    )
    if bool(bad_pid.any()):
        examples = _cell_examples(df.loc[bad_pid])
        raise AssertionError(
            "all rows must include a positive-integer worker_pid; "
            f"found {int(bad_pid.sum())} invalid row(s): {examples}"
        )

    parsed_times: dict[str, pd.Series] = {}
    for col in ("cell_started_utc", "cell_finished_utc"):
        text = df[col].fillna("").astype(str).str.strip()
        parsed = pd.to_datetime(text, errors="coerce", utc=True)
        parsed_times[col] = parsed
        bad_ts = text.eq("") | parsed.isna()
        if bool(bad_ts.any()):
            examples = _cell_examples(df.loc[bad_ts])
            raise AssertionError(
                f"all rows must include parseable UTC {col}; "
                f"found {int(bad_ts.sum())} invalid row(s): {examples}"
            )

    finished_before_started = (
        parsed_times["cell_finished_utc"] < parsed_times["cell_started_utc"]
    )
    if bool(finished_before_started.any()):
        examples = _cell_examples(df.loc[finished_before_started])
        raise AssertionError(
            "cell_finished_utc must not be earlier than cell_started_utc; "
            f"found {int(finished_before_started.sum())} invalid row(s): "
            f"{examples}"
        )

    bad_duration = invalid_runtime_duration_mask(
        df["t_total_s"],
        parsed_times["cell_started_utc"],
        parsed_times["cell_finished_utc"],
    )
    if bool(bad_duration.any()):
        examples = _cell_examples(df.loc[bad_duration])
        raise AssertionError(
            "t_total_s must agree with cell_finished_utc - cell_started_utc "
            "within the runtime trace tolerance; "
            f"found {int(bad_duration.sum())} invalid row(s): {examples}"
        )

    rss_values: dict[str, pd.Series] = {}
    for col in ("rss_gb_start", "rss_gb_end", "rss_gb_peak"):
        rss = pd.to_numeric(df[col], errors="coerce")
        rss_values[col] = rss
        bad_rss = rss.isna() | ~np.isfinite(rss) | rss.lt(0)
        if bool(bad_rss.any()):
            examples = _cell_examples(df.loc[bad_rss])
            raise AssertionError(
                f"all rows must include finite non-negative {col}; "
                f"found {int(bad_rss.sum())} invalid row(s): {examples}"
            )

    bad_peak_order = (
        rss_values["rss_gb_peak"].notna()
        & np.isfinite(rss_values["rss_gb_peak"])
        & (
            rss_values["rss_gb_peak"].lt(rss_values["rss_gb_start"])
            | rss_values["rss_gb_peak"].lt(rss_values["rss_gb_end"])
        )
    )
    if bool(bad_peak_order.any()):
        examples = _cell_examples(df.loc[bad_peak_order])
        raise AssertionError(
            "rss_gb_peak must be greater than or equal to rss_gb_start "
            f"and rss_gb_end; found {int(bad_peak_order.sum())} invalid "
            f"row(s): {examples}"
        )


def _missing_protocol_key_values(df: pd.DataFrame) -> list[str]:
    missing: list[str] = []
    for col in ["case_id", "method", "budget_mult", "repeat"]:
        text = df[col].fillna("").astype(str).str.strip()
        n_missing = int(text.eq("").sum())
        if n_missing:
            missing.append(f"{col}={n_missing}")
    return missing


def _cell_examples(df: pd.DataFrame, *, limit: int = 3) -> list[dict[str, object]]:
    cols = [
        col
        for col in ["case_id", "method", "budget_mult", "repeat", "status"]
        if col in df.columns
    ]
    return df[cols].head(limit).to_dict(orient="records")


def declared_methods_from_manifest(manifest_path: Path) -> list[str]:
    """Return ``experiment.methods_declared`` from a RUN_MANIFEST.json."""
    if not manifest_path.exists():
        return []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AssertionError(
            f"could not parse RUN_MANIFEST.json for method validation: {exc}"
        ) from exc
    raw = manifest.get("experiment", {}).get("methods_declared")
    if not isinstance(raw, list):
        return []
    methods: list[str] = []
    for item in raw:
        if item is None:
            continue
        name = str(item).strip()
        if name:
            methods.append(name)
    return methods


# ─── Fig 1 ───────────────────────────────────────────────────────────────
def _method_catalog_rows(method_names: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name in method_names:
        if "+" in name:
            family, surrogate = name.split("+", 1)
        else:
            family, surrogate = name, name
        rows.append({
            "method": name,
            "family": family,
            "surrogate": surrogate,
            "pool": "nature",
        })
    return rows


def make_fig1(
    df: pd.DataFrame,
    out: Path,
    *,
    declared_methods: list[str] | None = None,
    method_source: str = "declared methods",
) -> None:
    """Fig 1: case catalog + method catalog."""
    import doe_testbed as dbt

    cases = []
    for c in dbt.list_cases():
        meta = getattr(c, "meta", c)
        topology = getattr(meta, "response_topology", None) or []
        if isinstance(topology, str):
            topology = [topology]
        cases.append({
            "case_id": getattr(c, "case_id", getattr(meta, "case_id", "?")),
            "domain": getattr(meta, "domain", "unknown"),
            "dim": getattr(meta, "dim_input", None),
            "n_output": getattr(meta, "dim_output", None),
            "topology": ";".join(str(t) for t in topology),
            "has_failure": bool(getattr(meta, "has_failure_boundary", False)),
            "supports_mf": bool(getattr(meta, "supports_multifidelity", False)),
        })
    _save(
        pd.DataFrame(cases),
        out / "fig1_case_catalog.csv",
        "doe_testbed.list_cases()",
    )

    if declared_methods:
        rows = _method_catalog_rows(list(declared_methods))
        source = method_source
        # A declared method name identifies its sampler, not its method family.
        # Use the archived classification so manifest-driven reproduction agrees
        # with the original registry-driven catalog (e.g. lhs+gp: static_design).
        if "method_family" in df.columns:
            families = df[["method", "method_family"]].dropna().drop_duplicates()
            assert not families["method"].duplicated().any(), (
                "results.csv declares multiple families for the same method"
            )
            by_method = families.set_index("method")["method_family"].to_dict()
            for row in rows:
                if row["method"] in by_method:
                    row["family"] = by_method[row["method"]]
    else:
        from benchmark import nature_pool
        methods = nature_pool()
        rows = []
        for m in methods:
            name = m.name
            if "+" in name:
                family, surrogate = name.split("+", 1)
            else:
                family, surrogate = name, name
            rows.append({
                "method": name,
                "family": getattr(m, "family", family),
                "surrogate": surrogate,
                "pool": "nature",
            })
        source = "benchmark.methods.nature_pool()"
    _save(
        pd.DataFrame(rows),
        out / "fig1_method_catalog.csv",
        source,
    )


# ─── Fig 2 ───────────────────────────────────────────────────────────────
def make_fig2(d1_dir: Path, out: Path, *, allow_legacy_rank: bool = False) -> None:
    """Fig 2: D1 Spearman ρ + rank comparison scatter."""
    corr_json = d1_dir / "correlation.json"
    assert corr_json.exists(), f"{corr_json} not found — run Stage 4 first"
    corr = pd.DataFrame(json.loads(corr_json.read_text(encoding="utf-8")))
    if "analysis_type" not in corr.columns:
        corr["analysis_type"] = D1_ANALYSIS
    if "is_primary" not in corr.columns:
        corr["is_primary"] = True
    _save(corr, out / "fig2_d1_spearman.csv", str(corr_json))

    parts = []
    for pairing_dir in sorted(d1_dir.glob("pairing_*")):
        csv_path = pairing_dir / "rank_comparison.csv"
        if not csv_path.exists():
            continue
        part = pd.read_csv(csv_path)
        mode = pairing_dir.name.replace("pairing_", "")
        if "pairing_mode" not in part.columns:
            part["pairing_mode"] = mode
        if "analysis_type" not in part.columns:
            part["analysis_type"] = D1_ANALYSIS
        if "is_primary" not in part.columns:
            part["is_primary"] = True
        parts.append(part)
    if not parts:
        legacy = d1_dir / "rank_comparison.csv"
        if legacy.exists() and allow_legacy_rank:
            part = pd.read_csv(legacy)
            if "pairing_mode" not in part.columns:
                part["pairing_mode"] = "dimension"
            if "analysis_type" not in part.columns:
                part["analysis_type"] = D1_ANALYSIS
            if "is_primary" not in part.columns:
                part["is_primary"] = True
            parts.append(part)
        elif legacy.exists():
            raise AssertionError(
                f"{legacy} is a legacy D1 layout; regenerate Stage 4 with "
                "pairing_* subdirectories or pass --allow-legacy-d1-rank for "
                "an explicit archive conversion"
            )
    assert parts, f"no nested pairing_*/rank_comparison.csv found under {d1_dir}"
    ranks = pd.concat(parts, ignore_index=True)
    assert "pairing_mode" in ranks.columns, "rank_comparison.csv missing pairing_mode"
    _save(
        ranks,
        out / "fig2_d1_ranks.csv",
        str(d1_dir / "pairing_*" / "rank_comparison.csv"),
    )


# ─── Fig 3 ───────────────────────────────────────────────────────────────
def make_fig3(d2_dir: Path, out: Path) -> None:
    """Fig 3: D2 learning curves + 10·d rule bar chart."""
    lc_csv = d2_dir / "learning_curves.csv"
    rs_csv = d2_dir / "rule_summary.csv"
    assert lc_csv.exists(), f"{lc_csv} not found — run Stage 5 first"
    assert rs_csv.exists(), f"{rs_csv} not found — run Stage 5 first"

    lc = pd.read_csv(lc_csv)
    assert "n_cases" in lc.columns, "learning_curves.csv missing n_cases"
    assert {"median", "ci_lo", "ci_hi"}.issubset(lc.columns), (
        "learning_curves.csv must contain median/ci_lo/ci_hi"
    )
    _save(lc, out / "fig3_d2_learning.csv", str(lc_csv))

    rs = pd.read_csv(rs_csv)
    assert "n_cases" in rs.columns, (
        "rule_summary.csv missing n_cases — fix GAP-11 first"
    )
    assert "exploratory" in rs.columns, (
        "rule_summary.csv missing exploratory — fix GAP-11 first"
    )
    if "analysis_type" not in rs.columns:
        rs["analysis_type"] = D2_ANALYSIS
    if "is_primary" not in rs.columns:
        rs["is_primary"] = rs["analysis_type"].astype(str).eq(D2_ANALYSIS)
    _save(rs, out / "fig3_d2_rule.csv", str(rs_csv))


# ─── Fig 4 ───────────────────────────────────────────────────────────────
def make_fig4(d3_dir: Path, out: Path) -> None:
    """Fig 4: D3 signed bias per method."""
    bias_csv = d3_dir / "reliability_per_method.csv"
    assert bias_csv.exists(), f"{bias_csv} not found — run Stage 6 first"
    bias = pd.read_csv(bias_csv)
    rename_map = {
        "signed_bias_mean": "signed_bias",
        "signed_bias_ci_lo": "ci_lo",
        "signed_bias_ci_hi": "ci_hi",
        "signed_bias_pvalue_two_sided": "p_raw_two_sided",
        "signed_bias_pvalue_neg": "p_raw_neg",
        "signed_bias_pvalue_pos": "p_raw_pos",
    }
    for src, dst in rename_map.items():
        if src in bias.columns and dst not in bias.columns:
            bias[dst] = bias[src]
    if "p_holm" not in bias.columns:
        # Current artifacts must carry the registered positive one-sided
        # p-value.  Keep a two-sided fallback only for explicitly legacy
        # tables used by compatibility tests; the submission gate requires
        # the positive column and therefore cannot accept this fallback.
        p_col = "p_raw_pos" if "p_raw_pos" in bias.columns else "p_raw_two_sided"
        if p_col in bias.columns:
            bias["p_holm"] = _holm_adjust(
                bias[p_col].to_numpy(dtype=float)
            )
    if "bias_direction" not in bias.columns and "signed_bias" in bias.columns:
        # NaN signed_bias means "no data" (e.g. method had no ok runs on the
        # 16 failure cases), not "no bias".  The original two-branch
        # ``np.where`` collapsed NaN into the "none" bucket alongside
        # exact-zero — those are semantically different and the §D3
        # decision rule counts them differently.  We carve out a
        # ``"undefined"`` label for NaN inputs and reserve ``"none"`` for
        # finite-but-zero bias.
        sb = bias["signed_bias"].to_numpy(dtype=float)
        direction = np.where(
            np.isnan(sb), "undefined",
            np.where(sb > 0, "over", np.where(sb < 0, "under", "none")),
        )
        bias["bias_direction"] = direction
    if "low_n" not in bias.columns:
        if "n_runs" in bias.columns:
            bias["low_n"] = pd.to_numeric(
                bias["n_runs"], errors="coerce",
            ).lt(10)
        else:
            bias["low_n"] = False
    _save(bias, out / "fig4_d3_bias.csv", str(bias_csv))


# ─── Fig 5 ───────────────────────────────────────────────────────────────
def make_fig5(df: pd.DataFrame, out: Path) -> None:
    """Fig 5: Mechanism heatmap — stratified ranks by topology."""
    from benchmark.stratified import annotate_results, stratified_ranks

    df_ann = annotate_results(df)
    largest_budget = int(df_ann["budget_mult"].max())
    sr_wide = stratified_ranks(
        df_ann, stratify_by="topology", budget_value=largest_budget,
    )
    meta_cols = {
        "stratum", "n_cases", "friedman_chi2", "friedman_p", "critical_difference",
    }
    method_cols = [c for c in sr_wide.columns if c not in meta_cols]
    sr_long = sr_wide.melt(
        id_vars=[c for c in sr_wide.columns if c in meta_cols],
        value_vars=method_cols,
        var_name="method",
        value_name="mean_rank",
    ).rename(columns={"stratum": "topology"})
    sr_long["budget_mult"] = largest_budget
    if "n_cases" in sr_long.columns:
        sr_long["exploratory"] = sr_long["n_cases"] < 3
    _save(
        sr_long,
        out / "fig5_mechanism.csv",
        "benchmark.stratified.stratified_ranks()",
    )


# ─── Fig 6 ───────────────────────────────────────────────────────────────
def make_fig6(df: pd.DataFrame, out: Path) -> None:
    """Fig 6: Pareto front — NRMSE vs compute time."""
    from benchmark.rigour import pareto_front

    cost_col = next(
        (
            c
            for c in [
                "t_total_s",
                "wall_time_s",
                "wallclock_s",
                "t_predict_s",
                "t_fit_s",
            ]
            if c in df.columns
        ),
        None,
    )
    assert cost_col is not None, (
        "results.csv has no timing column for Pareto analysis"
    )

    rows = []
    for bm in sorted(df["budget_mult"].dropna().unique()):
        bm = int(bm)
        sub_all = df[df["budget_mult"] == bm].copy()
        sub_ok = sub_all[sub_all["status"] == "ok"].copy()
        if sub_all.empty or sub_ok.empty:
            continue
        denom = (
            sub_all.groupby("method")
            .agg(
                n_total=("method", "size"),
                n_ok=("status", lambda s: int((s == "ok").sum())),
            )
            .reset_index()
        )
        denom["failure_rate"] = 1.0 - denom["n_ok"] / denom["n_total"]

        # Complete-case Pareto.
        agg = (
            sub_ok.groupby("method")
            .agg(
                nrmse_median=("nrmse", "median"),
                time_median_s=(cost_col, "median"),
                time_mean_s=(cost_col, "mean"),
            )
            .reset_index()
        )
        agg = agg.merge(denom, on="method", how="left")
        agg["budget_mult"] = bm
        agg["analysis_type"] = "complete_case"
        agg["is_primary"] = False
        try:
            pa = pareto_front(
                sub_ok, metric="nrmse", cost_col=cost_col, budget_value=bm,
            )
            on_pareto = set(pa[pa["on_pareto"]]["method"].tolist())
        except Exception:
            on_pareto = set()
        agg["on_pareto"] = agg["method"].isin(on_pareto)
        rows.append(agg)

        # Penalized Pareto: failed/timeout retained in the denominator.
        penalty = float(np.nanmax(sub_ok["nrmse"])) * 1.25
        if not np.isfinite(penalty) or penalty <= 0:
            penalty = 10.0
        sub_pen = sub_all.copy()
        sub_pen["nrmse_penalized"] = (
            sub_pen["nrmse"].where(sub_pen["status"] == "ok", penalty).fillna(penalty)
        )
        pen = (
            sub_pen.groupby("method")
            .agg(
                nrmse_median=("nrmse_penalized", "median"),
                time_median_s=(cost_col, "median"),
                time_mean_s=(cost_col, "mean"),
            )
            .reset_index()
        )
        pen = pen.merge(denom, on="method", how="left")
        pen["budget_mult"] = bm
        pen["analysis_type"] = "penalized_failed_in_denominator"
        pen["is_primary"] = True
        try:
            pa_pen = pareto_front(
                sub_pen,
                metric="nrmse_penalized",
                cost_col=cost_col,
                budget_value=bm,
            )
            on_pen = set(pa_pen[pa_pen["on_pareto"]]["method"].tolist())
        except Exception:
            on_pen = set()
        pen["on_pareto"] = pen["method"].isin(on_pen)
        rows.append(pen)

    pareto_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    _save(
        pareto_df,
        out / "fig6_pareto.csv",
        "benchmark.rigour.pareto_front() on results.csv",
    )


# ─── CLI ────────────────────────────────────────────────────────────────
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "Build the figdata package consumed by plots_nature.py "
            "(GAP-12)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--results-csv",
        type=Path,
        default=ROOT / "benchmark" / "results" / "full_benchmark" / "results.csv",
    )
    p.add_argument(
        "--d1-dir",
        type=Path,
        default=ROOT / "benchmark" / "results" / "ranking_reversal",
    )
    p.add_argument(
        "--d2-dir",
        type=Path,
        default=ROOT / "benchmark" / "results" / "sample_complexity_0.20",
    )
    p.add_argument(
        "--d3-dir",
        type=Path,
        default=ROOT / "benchmark" / "results" / "reliability_recovery",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=FIGDATA_DIR,
    )
    p.add_argument(
        "--expected-methods",
        default=None,
        help=(
            "Comma-separated declared method names. If provided, Stage 8 "
            "fails before writing figdata when results.csv contains extras."
        ),
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "RUN_MANIFEST.json whose experiment.methods_declared should be "
            "used for Stage 8 method validation. Takes precedence over "
            "--expected-methods."
        ),
    )
    p.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["fig1", "fig2", "fig3", "fig4", "fig5", "fig6"],
        help="Skip selected figures (useful when a sub-stage is still running).",
    )
    p.add_argument(
        "--allow-legacy-d1-rank",
        action="store_true",
        help=(
            "Permit the pre-pairing root d1-dir/rank_comparison.csv layout. "
            "Formal paper runs should regenerate Stage 4 and leave this off."
        ),
    )
    p.add_argument(
        "--require-headline-source",
        action="store_true",
        help=(
            "Fail unless --results-csv is the canonical full_benchmark/results.csv "
            "and the CSV is the exact frozen 34 x 20 x 4 x 30 nature-pool grid. "
            "Use this for formal paper figdata; leave it off for small fixtures."
        ),
    )
    return p.parse_args(argv)


def _build_figdata(
    args: argparse.Namespace,
    df: pd.DataFrame,
    out_dir: Path,
    expected_methods: list[str],
    expected_method_source: str,
) -> None:
    if "fig1" not in args.skip:
        print("\n[Fig 1] Case + method catalog...")
        make_fig1(
            df,
            out_dir,
            declared_methods=expected_methods,
            method_source=expected_method_source or "declared methods",
        )
    if "fig2" not in args.skip:
        print("\n[Fig 2] D1 Spearman ρ + rank scatter...")
        make_fig2(args.d1_dir, out_dir, allow_legacy_rank=args.allow_legacy_d1_rank)
    if "fig3" not in args.skip:
        print("\n[Fig 3] D2 learning curves + 10·d rule...")
        make_fig3(args.d2_dir, out_dir)
    if "fig4" not in args.skip:
        print("\n[Fig 4] D3 reliability signed bias...")
        make_fig4(args.d3_dir, out_dir)
    if "fig5" not in args.skip:
        print("\n[Fig 5] Mechanism: stratified ranks by topology...")
        make_fig5(df, out_dir)
    if "fig6" not in args.skip:
        print("\n[Fig 6] Pareto front: NRMSE vs wall_time...")
        make_fig6(df, out_dir)


def _replace_directory_atomically(src: Path, dst: Path) -> None:
    """Replace a derived figdata directory only after a complete build exists."""
    dst = dst.resolve()
    src = src.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    backup: Path | None = None
    if dst.exists():
        backup = dst.with_name(f".{dst.name}.backup.{_ts().replace(':', '').replace('+', 'Z')}")
        if backup.exists():
            shutil.rmtree(backup)
        dst.rename(backup)
    try:
        shutil.move(str(src), str(dst))
    except Exception:
        if dst.exists():
            shutil.rmtree(dst)
        if backup is not None and backup.exists():
            backup.rename(dst)
        raise
    if backup is not None and backup.exists():
        shutil.rmtree(backup)


def _manifest_entry(path: Path, *, label: str) -> dict[str, object]:
    entry: dict[str, object] = {
        "label": label,
        "path": portable_path(path, ROOT),
        "exists": path.exists(),
    }
    if path.exists() and path.is_file():
        entry.update({
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    return entry


def _analysis_inputs(args: argparse.Namespace) -> list[dict[str, object]]:
    inputs = [
        _manifest_entry(args.results_csv, label="results_csv"),
    ]
    if args.manifest is not None:
        inputs.append(_manifest_entry(args.manifest, label="run_manifest"))
    inputs.append(_manifest_entry(args.d1_dir / "correlation.json", label="d1_correlation"))
    for rank_csv in sorted(args.d1_dir.glob("pairing_*/rank_comparison.csv")):
        inputs.append(_manifest_entry(rank_csv, label="d1_rank_comparison"))
    legacy_rank = args.d1_dir / "rank_comparison.csv"
    if legacy_rank.exists():
        inputs.append(_manifest_entry(legacy_rank, label="d1_legacy_rank_comparison"))
    inputs.extend([
        _manifest_entry(args.d2_dir / "learning_curves.csv", label="d2_learning_curves"),
        _manifest_entry(args.d2_dir / "rule_summary.csv", label="d2_rule_summary"),
        _manifest_entry(args.d3_dir / "reliability_per_method.csv", label="d3_reliability_per_method"),
    ])
    return inputs


def _results_csv_summary(results_csv: Path) -> dict[str, object]:
    """Small machine-readable summary for figdata provenance checks."""
    if not results_csv.exists() or not results_csv.is_file():
        return {"exists": False}
    df = pd.read_csv(results_csv, low_memory=False)
    summary: dict[str, object] = {
        "exists": True,
        "n_rows": int(len(df)),
    }
    for column, key in (
        ("case_id", "n_cases"),
        ("method", "n_methods"),
    ):
        if column in df.columns:
            summary[key] = int(df[column].dropna().astype(str).nunique())
    if "budget_mult" in df.columns:
        summary["budgets"] = [
            int(v) for v in sorted(_integer_series(df["budget_mult"], "budget_mult").unique())
        ]
    if "repeat" in df.columns:
        summary["repeats"] = [
            int(v) for v in sorted(_integer_series(df["repeat"], "repeat").unique())
        ]
    if "status" in df.columns:
        summary["status_counts"] = {
            str(k): int(v)
            for k, v in df["status"].fillna("").astype(str).value_counts().sort_index().items()
        }
    if "failure_category" in df.columns:
        summary["failure_category_counts"] = {
            str(k): int(v)
            for k, v in (
                df["failure_category"]
                .fillna("")
                .astype(str)
                .value_counts()
                .sort_index()
                .items()
            )
        }
    return summary


def write_figdata_manifest(
    args: argparse.Namespace,
    out_dir: Path,
    *,
    status: str = "complete",
    extra_errors: list[str] | None = None,
) -> None:
    """Record source-input and derived-CSV hashes for the figdata package."""
    figdata_files = [
        _manifest_entry(path, label="figdata_csv")
        for path in sorted(out_dir.glob("*.csv"))
    ]
    payload = {
        "generated_at": _ts(),
        "script": "scripts/prepare_figdata.py",
        "status": status,
        "results_csv": portable_path(args.results_csv, ROOT),
        "d1_dir": portable_path(args.d1_dir, ROOT),
        "d2_dir": portable_path(args.d2_dir, ROOT),
        "d3_dir": portable_path(args.d3_dir, ROOT),
        "analysis_policy": ANALYSIS_POLICY,
        "results_summary": _results_csv_summary(args.results_csv),
        "inputs": _analysis_inputs(args),
        "figdata_files": figdata_files,
        "errors": (extra_errors or []) + [
            f"missing input: {item['label']} {item['path']}"
            for item in _analysis_inputs(args)
            if not item.get("exists")
        ],
    }
    (out_dir / "figdata_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main(argv=None):
    """Derive figure-source CSVs from the frozen sweep and analysis artifacts."""
    args = parse_args(argv)

    print(f"Loading results.csv from {args.results_csv}...")
    df = pd.read_csv(args.results_csv, low_memory=False)
    print(
        f"  {len(df)} rows, {df['method'].nunique()} methods, "
        f"{df['case_id'].nunique()} cases"
    )
    validate_results_contract(df)
    expected_methods: list[str] = []
    expected_method_source = ""
    if args.manifest is not None:
        expected_methods = declared_methods_from_manifest(args.manifest)
        if expected_methods:
            expected_method_source = "RUN_MANIFEST experiment.methods_declared"
    if not expected_methods:
        expected_methods = _split_csv_arg(args.expected_methods)
        if expected_methods:
            expected_method_source = "--expected-methods"
    validate_results_methods(df, expected_methods)
    if args.require_headline_source:
        validate_headline_results_source(
            args.results_csv,
            df,
            declared_methods=expected_methods,
        )

    if args.skip:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        print(
            "\n[warn] --skip requested; writing selected figdata directly "
            "without full-package replacement."
        )
        _build_figdata(args, df, args.out_dir, expected_methods, expected_method_source)
        write_figdata_manifest(
            args,
            args.out_dir,
            status="partial_skip_selection",
            extra_errors=[
                "partial figdata package produced with --skip; rerun without "
                "--skip before submission"
            ],
        )
    else:
        tmp_parent = Path(tempfile.mkdtemp(
            prefix=f".{args.out_dir.name}.tmp.",
            dir=str(args.out_dir.resolve().parent),
        ))
        tmp_out = tmp_parent / args.out_dir.name
        try:
            _build_figdata(args, df, tmp_out, expected_methods, expected_method_source)
            write_figdata_manifest(args, tmp_out)
            _replace_directory_atomically(tmp_out, args.out_dir)
        finally:
            if tmp_parent.exists():
                shutil.rmtree(tmp_parent)

    print(f"\n[done] All figdata written to {args.out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
