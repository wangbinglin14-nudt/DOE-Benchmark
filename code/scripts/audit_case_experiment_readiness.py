"""audit_case_experiment_readiness.py
=====================================

Lightweight all-case audit for benchmark-run suitability.

The physics tests and case-validation ledger answer whether each case is
physically plausible.  This script answers a narrower operational question:
can every registered case be sampled, evaluated in batch, and represented in
the full-benchmark result metadata without surprising shape, NaN, timing, or
failure-threshold issues?
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import doe_testbed as db


ALLOWED_RESPONSE_TOPOLOGY = {
    "smooth_monotone",
    "smooth_unimodal",
    "threshold_jump",
    "discontinuous",
    "strong_interaction",
    "sparse_effects",
    "constrained_domain",
    "multi_modal",
    "smooth_multimodal",
    "heterogeneous",
}
ALLOWED_DOMAINS = {
    "aerodynamics",
    "structures",
    "heat_transfer",
    "chemical_process",
    "chemical",
    "geomechanics",
    "multidisciplinary",
}
ALLOWED_PDE_TYPES = {
    "elliptic",
    "parabolic",
    "hyperbolic",
    "ode_system",
    "algebraic",
    "integral",
    "hybrid",
    "analytical",
    "unknown",
}
REQUIRED_COLUMNS = [
    "case_id",
    "domain",
    "pde_type",
    "response_topology",
    "n_topology_tags",
    "governing_equation",
    "n_physics_references",
    "dim",
    "n_output",
    "n_samples",
    "status",
    "elapsed_ms",
    "ms_per_sample",
    "n_nan",
    "n_inf",
    "input_mutation_max_abs_delta",
    "input_mutation_n_changed",
    "determinism_max_abs_delta",
    "determinism_n_mismatch",
    "n_expected_range_checked",
    "n_expected_range_violations",
    "expected_range_max_abs_violation",
    "has_failure_boundary",
    "n_failure_thresholds",
    "supports_multifidelity",
    "fidelity_levels",
    "message",
]


def _failure_threshold_count(case: Any) -> int:
    return sum(
        1
        for output in getattr(case.meta, "output_variables", [])
        if getattr(output, "failure_threshold", None) is not None
    )


def _one_line(value: Any) -> str:
    return " ".join(str(value or "").split())


def _metadata_errors(case: Any) -> list[str]:
    meta = case.meta
    errors: list[str] = []
    if not str(getattr(meta, "case_id", "")).strip():
        errors.append("case_id must be non-empty")
    if not str(getattr(meta, "name", "")).strip():
        errors.append("name must be non-empty")
    domain = str(getattr(meta, "domain", "")).strip()
    if domain not in ALLOWED_DOMAINS:
        errors.append(f"domain {domain!r} is not in the closed vocabulary")
    pde_type = str(getattr(meta, "pde_type", "")).strip()
    if pde_type not in ALLOWED_PDE_TYPES:
        errors.append(f"pde_type {pde_type!r} is not in the closed vocabulary")
    if not str(getattr(meta, "governing_equation", "")).strip():
        errors.append("governing_equation must be non-empty")
    if not getattr(meta, "physics_references", []):
        errors.append("physics_references must be non-empty")
    topology = list(getattr(meta, "response_topology", []) or [])
    if not topology:
        errors.append("response_topology must contain at least one tag")
    unknown_topology = sorted(set(str(item) for item in topology) - ALLOWED_RESPONSE_TOPOLOGY)
    if unknown_topology:
        errors.append(f"unknown response_topology tags {unknown_topology}")
    if len(topology) != len(set(topology)):
        errors.append(f"duplicate response_topology tags {topology!r}")
    if int(meta.dim_input) <= 0:
        errors.append("dim_input must be positive")
    if int(meta.dim_output) <= 0:
        errors.append("dim_output must be positive")
    input_variables = list(getattr(meta, "input_variables", []) or [])
    output_variables = list(getattr(meta, "output_variables", []) or [])
    if len(input_variables) != int(meta.dim_input):
        errors.append("input_variables count does not match dim_input")
    if len(output_variables) != int(meta.dim_output):
        errors.append("output_variables count does not match dim_output")
    input_names = [str(getattr(item, "name", "")).strip() for item in input_variables]
    output_names = [str(getattr(item, "name", "")).strip() for item in output_variables]
    if len(input_names) != len(set(input_names)):
        errors.append(f"duplicate input variable names {input_names!r}")
    if len(output_names) != len(set(output_names)):
        errors.append(f"duplicate output variable names {output_names!r}")
    for var in input_variables:
        name = str(getattr(var, "name", "")).strip() or "<input>"
        if not name or name == "<input>":
            errors.append("input variable name must be non-empty")
        if not str(getattr(var, "description", "")).strip():
            errors.append(f"{name}: input description must be non-empty")
        if not str(getattr(var, "unit", "")).strip():
            errors.append(f"{name}: input unit must be non-empty")
        try:
            lower = float(getattr(var, "lower", math.nan))
            upper = float(getattr(var, "upper", math.nan))
            if not lower < upper:
                errors.append(f"{name}: input lower must be < upper")
            if getattr(var, "log_scale", False) and lower <= 0:
                errors.append(f"{name}: log_scale input lower bound must be positive")
        except (TypeError, ValueError):
            errors.append(f"{name}: input bounds must be numeric")
    for output in output_variables:
        name = str(getattr(output, "name", "")).strip() or "<output>"
        if not name or name == "<output>":
            errors.append("output variable name must be non-empty")
        if not str(getattr(output, "description", "")).strip():
            errors.append(f"{name}: output description must be non-empty")
        if not str(getattr(output, "unit", "")).strip():
            errors.append(f"{name}: output unit must be non-empty")
        expected_range = getattr(output, "expected_range", None)
        if expected_range is not None:
            try:
                lower, upper = expected_range
                lower_f = float(lower)
                upper_f = float(upper)
                if not lower_f < upper_f:
                    errors.append(f"{name}: expected_range lower must be < upper")
                threshold = getattr(output, "failure_threshold", None)
                if threshold is not None:
                    threshold_f = float(threshold)
                    if not lower_f <= threshold_f <= upper_f:
                        errors.append(
                            f"{name}: failure_threshold {threshold_f:.6g} "
                            f"outside expected_range [{lower_f:.6g}, {upper_f:.6g}]"
                        )
            except (TypeError, ValueError):
                errors.append(f"{name}: expected_range must be a numeric (lower, upper) pair")
    if getattr(meta, "has_failure_boundary", False) and _failure_threshold_count(case) == 0:
        errors.append("has_failure_boundary=True but no output failure_threshold")
    for output in output_variables:
        threshold = getattr(output, "failure_threshold", None)
        direction = getattr(output, "failure_direction", None)
        if threshold is not None and direction not in {"above", "below"}:
            errors.append(
                f"{getattr(output, 'name', '<output>')}: invalid failure_direction {direction!r}"
            )
    return errors


def _envelope_probe_points(case: Any) -> np.ndarray:
    bounds = [(var.lower, var.upper) for var in case.meta.input_variables]
    if len(bounds) <= 12:
        return np.array(list(itertools.product(*bounds)), dtype=float)

    midpoint = np.array([(lo + hi) / 2.0 for lo, hi in bounds], dtype=float)
    points = [midpoint.copy()]
    for index, (lower, upper) in enumerate(bounds):
        for value in (lower, upper):
            point = midpoint.copy()
            point[index] = value
            points.append(point)
    return np.array(points, dtype=float)


def _range_tolerance(lower: float, upper: float) -> float:
    return 1e-9 * max(1.0, abs(lower), abs(upper))


def _expected_range_envelope_errors(case: Any, Y: np.ndarray) -> tuple[int, int, float, list[str]]:
    errors: list[str] = []
    n_checked = 0
    n_violations = 0
    max_abs_violation = 0.0
    if not np.all(np.isfinite(Y)):
        errors.append("expected_range envelope produced non-finite outputs")
        return n_checked, n_violations, math.inf, errors
    for index, output in enumerate(getattr(case.meta, "output_variables", [])):
        expected_range = getattr(output, "expected_range", None)
        if expected_range is None:
            continue
        n_checked += 1
        try:
            lower, upper = expected_range
            lower_f = float(lower)
            upper_f = float(upper)
        except (TypeError, ValueError):
            errors.append(
                f"{getattr(output, 'name', '<output>')}: malformed expected_range {expected_range!r}"
            )
            n_violations += 1
            max_abs_violation = math.inf
            continue
        values = Y[:, index]
        observed_lower = float(np.nanmin(values))
        observed_upper = float(np.nanmax(values))
        tolerance = _range_tolerance(lower_f, upper_f)
        lower_violation = max(0.0, lower_f - observed_lower - tolerance)
        upper_violation = max(0.0, observed_upper - upper_f - tolerance)
        violation = max(lower_violation, upper_violation)
        if violation > 0.0:
            n_violations += 1
            max_abs_violation = max(max_abs_violation, violation)
            errors.append(
                f"{getattr(output, 'name', '<output>')}: envelope observed "
                f"[{observed_lower:.6g}, {observed_upper:.6g}] outside "
                f"expected_range [{lower_f:.6g}, {upper_f:.6g}]"
            )
    return n_checked, n_violations, max_abs_violation, errors


def audit_case(case_id: str, *, n_samples: int, seed: int) -> dict[str, Any]:
    case = db.get_case(case_id)
    meta = case.meta
    response_topology = list(getattr(meta, "response_topology", []) or [])
    physics_references = list(getattr(meta, "physics_references", []) or [])
    row: dict[str, Any] = {
        "case_id": getattr(meta, "case_id", case_id),
        "domain": getattr(meta, "domain", ""),
        "pde_type": getattr(meta, "pde_type", ""),
        "response_topology": ";".join(str(item) for item in response_topology),
        "n_topology_tags": len(response_topology),
        "governing_equation": _one_line(getattr(meta, "governing_equation", "")),
        "n_physics_references": len(physics_references),
        "dim": int(meta.dim_input),
        "n_output": int(meta.dim_output),
        "n_samples": int(n_samples),
        "status": "ok",
        "elapsed_ms": "",
        "ms_per_sample": "",
        "n_nan": "",
        "n_inf": "",
        "input_mutation_max_abs_delta": "",
        "input_mutation_n_changed": "",
        "determinism_max_abs_delta": "",
        "determinism_n_mismatch": "",
        "n_expected_range_checked": "",
        "n_expected_range_violations": "",
        "expected_range_max_abs_violation": "",
        "has_failure_boundary": bool(meta.has_failure_boundary),
        "n_failure_thresholds": _failure_threshold_count(case),
        "supports_multifidelity": bool(meta.supports_multifidelity),
        "fidelity_levels": ";".join(str(item) for item in meta.fidelity_levels),
        "message": "",
    }
    messages = _metadata_errors(case)
    rng = np.random.default_rng(seed)
    # Include corners and centre because boundary transforms often expose
    # denormalisation, log-scale, and singular-formula mistakes; add random
    # points so vectorised batch evaluation is also exercised.
    base = [
        np.zeros((1, meta.dim_input)),
        np.ones((1, meta.dim_input)),
        np.full((1, meta.dim_input), 0.5),
    ]
    n_random = max(0, int(n_samples) - len(base))
    if n_random:
        base.append(rng.random((n_random, meta.dim_input)))
    X = np.vstack(base)
    X_before = X.copy()

    try:
        t0 = time.perf_counter()
        Y = case.evaluate(X)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        Y_repeat = case.evaluate(X)
    except Exception as exc:
        row["status"] = "error"
        row["message"] = "; ".join(messages + [f"{type(exc).__name__}: {exc}"])
        return row

    row["elapsed_ms"] = round(float(elapsed_ms), 3)
    row["ms_per_sample"] = round(float(elapsed_ms) / max(len(X), 1), 3)
    input_delta = np.abs(X - X_before)
    finite_input_delta = input_delta[np.isfinite(input_delta)]
    max_input_delta = (
        float(finite_input_delta.max()) if finite_input_delta.size else math.inf
    )
    n_input_changed = int(
        np.sum(~np.isclose(X, X_before, rtol=0.0, atol=1e-15, equal_nan=True))
    )
    row["input_mutation_max_abs_delta"] = (
        round(max_input_delta, 15) if math.isfinite(max_input_delta) else "inf"
    )
    row["input_mutation_n_changed"] = n_input_changed
    if n_input_changed:
        messages.append(
            "evaluate mutated its input array: "
            f"max_abs_delta={max_input_delta:.6g}, n_changed={n_input_changed}"
        )
    Y = np.asarray(Y, dtype=float)
    if Y.shape != (len(X), meta.dim_output):
        messages.append(f"evaluate returned shape {Y.shape}, expected {(len(X), meta.dim_output)}")
    try:
        Y_repeat = np.asarray(Y_repeat, dtype=float)
    except Exception as exc:
        messages.append(f"repeat evaluate returned non-numeric output: {type(exc).__name__}: {exc}")
        Y_repeat = np.empty((0, 0), dtype=float)
    if Y_repeat.shape != Y.shape:
        messages.append(f"repeat evaluate returned shape {Y_repeat.shape}, expected {Y.shape}")
    elif Y.size:
        delta = np.abs(Y_repeat - Y)
        finite_delta = delta[np.isfinite(delta)]
        max_delta = float(finite_delta.max()) if finite_delta.size else math.inf
        n_mismatch = int(np.sum(~np.isclose(Y_repeat, Y, rtol=0.0, atol=1e-12, equal_nan=True)))
        row["determinism_max_abs_delta"] = round(max_delta, 12) if math.isfinite(max_delta) else "inf"
        row["determinism_n_mismatch"] = n_mismatch
        if n_mismatch:
            messages.append(
                f"repeat evaluate is not deterministic: max_abs_delta={max_delta:.6g}, "
                f"n_mismatch={n_mismatch}"
            )
    else:
        row["determinism_max_abs_delta"] = 0.0
        row["determinism_n_mismatch"] = 0
    n_nan = int(np.isnan(Y).sum())
    n_inf = int(np.isinf(Y).sum())
    row["n_nan"] = n_nan
    row["n_inf"] = n_inf
    if n_nan or n_inf:
        messages.append(f"non-finite outputs: n_nan={n_nan}, n_inf={n_inf}")
    try:
        X_env = case.normalize(_envelope_probe_points(case))
        Y_env = np.asarray(case.evaluate(X_env), dtype=float)
        if Y_env.shape != (len(X_env), meta.dim_output):
            messages.append(
                f"expected_range envelope returned shape {Y_env.shape}, "
                f"expected {(len(X_env), meta.dim_output)}"
            )
        else:
            (
                n_expected_range_checked,
                n_expected_range_violations,
                expected_range_max_abs_violation,
                range_errors,
            ) = _expected_range_envelope_errors(case, Y_env)
            row["n_expected_range_checked"] = n_expected_range_checked
            row["n_expected_range_violations"] = n_expected_range_violations
            row["expected_range_max_abs_violation"] = (
                round(expected_range_max_abs_violation, 12)
                if math.isfinite(expected_range_max_abs_violation)
                else "inf"
            )
            messages.extend(range_errors)
    except Exception as exc:
        messages.append(f"expected_range envelope check failed: {type(exc).__name__}: {exc}")
    if not np.all(np.isfinite(Y)):
        row["status"] = "nonfinite"
    if row["determinism_n_mismatch"] not in {"", 0} and row["status"] == "ok":
        row["status"] = "nondeterministic"
    if row["input_mutation_n_changed"] not in {"", 0} and row["status"] == "ok":
        row["status"] = "input_mutation"
    if row["n_expected_range_violations"] not in {"", 0} and row["status"] == "ok":
        row["status"] = "metadata_warning"
    if messages and row["status"] == "ok":
        row["status"] = "metadata_warning"
    row["message"] = "; ".join(messages)

    typical = meta.typical_eval_time_ms.get("default", None)
    if typical is not None:
        try:
            typical_f = float(typical)
        except (TypeError, ValueError):
            typical_f = math.nan
        if math.isfinite(typical_f) and typical_f > 0:
            limit = max(typical_f * 50.0, 10.0)
            if float(row["ms_per_sample"]) > limit:
                msg = (
                    f"slow smoke evaluate: {row['ms_per_sample']} ms/sample "
                    f"> max(50*declared typical {typical_f}, 10 ms)"
                )
                row["message"] = "; ".join(x for x in [row["message"], msg] if x)
                if row["status"] == "ok":
                    row["status"] = "timing_warning"
    return row


def audit_all_cases(*, n_samples: int = 8, seed: int = 20260602) -> list[dict[str, Any]]:
    rows = []
    for meta in sorted(db.list_cases(), key=lambda item: item.case_id):
        rows.append(audit_case(meta.case_id, n_samples=n_samples, seed=seed))
    return rows


def write_rows(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_parent = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp.", dir=str(output.parent)))
    tmp_output = tmp_parent / output.name
    try:
        with tmp_output.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        shutil.move(str(tmp_output), str(output))
    finally:
        if tmp_parent.exists():
            shutil.rmtree(tmp_parent)


def summarize_statuses(
    rows: list[dict[str, Any]],
    *,
    fail_on_warning: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fatal = [row for row in rows if row["status"] not in {"ok", "timing_warning"}]
    warnings = [row for row in rows if row["status"] == "timing_warning"]
    if fail_on_warning:
        fatal.extend(warnings)
    return fatal, warnings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit all registered simulation cases for benchmark-run suitability.",
    )
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "case_experiment_readiness.csv",
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help=(
            "Treat timing_warning as a failing preflight condition. "
            "The formal full pipeline enables this before launching long sweeps."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.n_samples < 3:
        raise SystemExit("--n-samples must be at least 3 to include lower/upper/centre points")
    rows = audit_all_cases(n_samples=args.n_samples, seed=args.seed)
    write_rows(rows, args.output)
    bad, warnings = summarize_statuses(rows, fail_on_warning=args.fail_on_warning)
    print(f"[case-readiness] wrote {args.output} ({len(rows)} cases)")
    for row in rows:
        print(
            f"  {row['case_id']}: {row['status']} "
            f"({row['ms_per_sample']} ms/sample)"
            + (f" - {row['message']}" if row["message"] else "")
        )
    if bad:
        print(f"[result] FAIL - {len(bad)} case(s) are not experiment-ready.")
        if args.fail_on_warning and warnings:
            print(
                f"[result] fail-on-warning promoted {len(warnings)} "
                "timing warning(s) to failures."
            )
        return 1
    print("[result] PASS - all registered cases are experiment-ready for smoke evaluation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
