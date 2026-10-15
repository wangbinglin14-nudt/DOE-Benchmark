"""
scripts/fill_paper_numbers.py
==============================
从实验结果中提取所有论文数字，默认写入 ../manuscripts/numbers.tex 和 numbers.json。
所有数字来自 figdata / D1-D3 分析结果，不手动填写。

Usage::

    python scripts/fill_paper_numbers.py \
        --results-csv benchmark/results/full_benchmark/results.csv \
        --rr-dir      benchmark/results/ranking_reversal \
        --d2-dir      benchmark/results/sample_complexity_0.20 \
        --d3-dir      benchmark/results/reliability_recovery \
        --figdata-dir ../manuscripts/figdata \
        --output-tex  ../manuscripts/numbers.tex \
        --output-json ../manuscripts/numbers.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.project_layout import FIGDATA_DIR, MANUSCRIPTS_DIR, portable_path  # noqa: E402

import numpy as np
import pandas as pd

D1_RANK_POLICY = "complete_case"
D1_FALLBACK_POLICY = "complete_case"
D2_PRIMARY_ANALYSIS = "penalized_failed_in_denominator"
D3_MIN_RUNS_PER_METHOD = 10
HOLM_FWER_ALPHA = 0.05
HOLM_DISCOVERY_FAMILY_SIZE = 3
SUBMISSION_REQUIRED_NON_NA = {
    "totalRows",
    "okRows",
    "timeoutRows",
    "failedRows",
    "okFrac",
    "nCasesCovered",
    "nMethodsCovered",
    "dOneRho",
    "dOneCILo",
    "dOneCIHi",
    "dOneKendallTau",
    "dOneKendallTauCILo",
    "dOneKendallTauCIHi",
    "dOneDecision",
    "dOneVerb",
    "dOneNMethods",
    "dOneCaseAxisCILo",
    "dOneCaseAxisCIHi",
    "dOneCaseAxisProbBelow",
    "dOneReliabilityAnalytical",
    "dOneReliabilityPhysical",
    "dOneReliabilityCeiling",
    "dOneNPhysicalCasesComplete",
    "dTwoNMainTopos",
    "dTwoNExploTopos",
    "dTwoFracPassMain",
    "dTwoDecision",
    "dTwoConfirmatoryDecision",
    "dTwoMedianNStar",
    "dTwoCPUpper",
    "dTwoAltMetricPass",
    "dTwoAltMetricNClasses",
    "dThreeNSigMethods",
    "dThreeNTested",
    "dThreeNLowN",
    "dThreeDecision",
    "dThreeDirection",
    "dThreeNRuns",
    "dThreePooledBias",
    "dThreePooledCILo",
    "dThreePooledCIHi",
    "dThreeBCALo",
    "dThreeBCAHi",
    "dThreeLocoMin",
    "dThreeLocoMax",
    "dThreeNCases",
    "dThreeNCasesPositive",
    "dThreeCaseCILo",
    "dThreeCaseCIHi",
    "dThreeCaseP",
    "dThreeCaseThreshold",
    "dThreeNClusterExclZero",
    "dThreeNClusterInclZero",
    "dThreeNCaseHolmSurvivors",
    "dThreeLooCaseCILo",
    "dThreeLooCaseCIHi",
    "dThreeTopCaseSharePct",
    "dThreeRefScalePf",
    "dThreePctOfReference",
    "mcNoiseNTest",
    "mcNoiseMedianRelWidthPct",
    "mcNoiseMaxRelWidthPct",
    "mcNoiseMaxAbsWidth",
    "mcNoiseNCases",
}


def _holm_thresholds(alpha: float, k: int) -> list[float]:
    """Return the Holm-Bonferroni step-down thresholds for ranks 1..k."""
    return [alpha / (k - rank + 1) for rank in range(1, k + 1)]


def _format_thresholds(values: list[float]) -> str:
    """Compact, stable text for LaTeX macros and numbers.json.

    Four decimal places with trailing zeros removed, so the step-down list
    is quoted at the same precision as the case-level Holm threshold macro
    (both render the first step as 0.0167).
    """
    return ", ".join(f"{v:.4f}".rstrip("0").rstrip(".") for v in values)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _artifact_entry(path: Path, *, label: str) -> dict[str, object]:
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


def numbers_provenance(
    results_csv: Path,
    rr_dir: Path,
    d2_dir: Path,
    d3_dir: Path,
    figdata_dir: Path,
    test_protocol_dir: Path | None,
) -> dict[str, object]:
    """Record source hashes for paper/numbers.json."""
    if test_protocol_dir is None:
        test_protocol_dir = results_csv.parent / "analysis"
    bias_figdata = figdata_dir / "fig4_d3_bias.csv"
    d3_bias = bias_figdata if bias_figdata.exists() else d3_dir / "reliability_per_method.csv"
    inputs = [
        _artifact_entry(results_csv, label="results_csv"),
        _artifact_entry(rr_dir / "correlation.json", label="d1_correlation"),
        _artifact_entry(d2_dir / "rule_summary.csv", label="d2_rule_summary"),
        _artifact_entry(d3_bias, label="d3_bias_numbers"),
        _artifact_entry(test_protocol_dir / "test_set_sufficiency.csv", label="test_set_sufficiency"),
    ]
    return {
        "script": "scripts/fill_paper_numbers.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs,
    }


def _safe_round(value: float, ndigits: int = 3) -> object:
    """Round a float for LaTeX inclusion; return ``"N/A"`` on NaN/inf.

    LaTeX would render ``nan`` literally and break compile; ``"N/A"``
    is the documented sentinel (DOE-Benchmark §4.4 paper-numbers contract)
    that downstream macros and reviewers can both recognise.
    """
    if value is None:
        return "N/A"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(f):
        return "N/A"
    return round(f, ndigits)


def _is_na_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().upper() in {"", "N/A", "NA", "NAN", "??"}
    return False


def submission_completeness_errors(nums: dict) -> list[str]:
    """Return missing draft sentinels that are not allowed in a paper freeze."""
    errors: list[str] = []
    for key in sorted(SUBMISSION_REQUIRED_NON_NA):
        if key not in nums:
            errors.append(f"missing submission-critical number: {key}")
        elif _is_na_value(nums[key]):
            errors.append(
                f"submission-critical number {key} is {nums[key]!r}; "
                "regenerate upstream analysis before writing paper numbers"
            )
    return errors


def primary_policy_input_errors(rr_dir: Path, d2_dir: Path) -> list[str]:
    """Validate preregistered primary-analysis rows before freezing numbers.

    ``collect_numbers`` keeps legacy fallbacks for lightweight local fixtures,
    but the formal paper pipeline must not derive headline numbers from
    non-primary or missing-policy artifacts.
    """
    errors: list[str] = []

    corr_json = rr_dir / "correlation.json"
    if not corr_json.exists():
        errors.append(f"D1 primary policy check missing {corr_json}")
    else:
        try:
            corr = json.loads(corr_json.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"D1 primary policy check could not parse {corr_json}: {type(exc).__name__}: {exc}")
            corr = []
        if not isinstance(corr, list):
            errors.append(f"D1 primary policy check expected list records in {corr_json}")
            corr = []
        primary_dimension_10 = []
        wrong_primary = []
        for row in corr:
            if not isinstance(row, dict):
                continue
            try:
                budget = int(float(str(row.get("budget_mult", ""))))
            except ValueError:
                continue
            is_primary = _truthy_value(row.get("is_primary"))
            analysis_type = str(row.get("analysis_type", "")).strip()
            if is_primary and analysis_type != D1_RANK_POLICY:
                wrong_primary.append(row)
            if (
                str(row.get("pairing_mode", "")).strip() == "dimension"
                and budget == 10
                and is_primary
                and analysis_type == D1_RANK_POLICY
            ):
                primary_dimension_10.append(row)
        if wrong_primary:
            errors.append(
                "D1 primary policy check found is_primary=True records outside "
                f"{D1_RANK_POLICY!r}"
            )
        if not primary_dimension_10:
            errors.append(
                "D1 primary policy check requires a dimension/budget=10 "
                f"{D1_RANK_POLICY!r} row with is_primary=True"
            )

    rule_csv = d2_dir / "rule_summary.csv"
    if not rule_csv.exists():
        errors.append(f"D2 primary policy check missing {rule_csv}")
    else:
        try:
            rule = pd.read_csv(rule_csv)
        except Exception as exc:
            errors.append(f"D2 primary policy check could not parse {rule_csv}: {type(exc).__name__}: {exc}")
            rule = pd.DataFrame()
        required = {"analysis_type", "is_primary", "exploratory"}
        missing = sorted(required - set(rule.columns))
        if missing:
            errors.append(f"D2 primary policy check missing columns in {rule_csv}: {missing}")
        elif not rule.empty:
            is_primary = _truthy_series(rule["is_primary"])
            analysis_type = rule["analysis_type"].astype(str).str.strip()
            exploratory = _truthy_series(rule["exploratory"])
            wrong_primary = is_primary & analysis_type.ne(D2_PRIMARY_ANALYSIS)
            primary = is_primary & analysis_type.eq(D2_PRIMARY_ANALYSIS) & ~exploratory
            if bool(wrong_primary.any()):
                errors.append(
                    "D2 primary policy check found is_primary=True rows outside "
                    f"{D2_PRIMARY_ANALYSIS!r}"
                )
            if not bool(primary.any()):
                errors.append(
                    "D2 primary policy check requires at least one non-exploratory "
                    f"{D2_PRIMARY_ANALYSIS!r} row with is_primary=True"
                )
    return errors


def _classify_direction(sig_bias: pd.Series) -> str:
    """Classify the sign-pattern of significant signed-bias values.

    ``np.sign`` on an empty array returns an empty array; ``.all()`` on
    that empty array silently returns ``True`` due to vacuous truth.
    Without an explicit length guard this would mis-classify a row that
    is statistically significant but whose ``signed_bias`` happens to be
    NaN (e.g. only the F-metric was finite, the P_f sign was lost) as
    ``"over"``.  Return ``"none"`` instead so the paper does not assert
    a biased direction without underlying data.
    """
    signs = np.sign(sig_bias.dropna().to_numpy())
    if signs.size == 0:
        return "none"
    if (signs > 0).all():
        return "over"
    if (signs < 0).all():
        return "under"
    return "mixed"


def _holm_adjust(pvals: np.ndarray) -> np.ndarray:
    """Step-down Holm-Bonferroni matching benchmark.reliability_recovery."""
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


def _truthy_series(series: pd.Series) -> pd.Series:
    """Return a bool Series for CSV booleans such as True/False/1/0."""
    if series.dtype == bool:
        return series.fillna(False).astype(bool)
    text = series.astype(str).str.strip().str.lower()
    return text.isin({"true", "1", "yes", "y"})


def _truthy_value(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _select_d1_primary_record(records: list[dict[str, object]]) -> dict[str, object] | None:
    """Pick the preregistered D1 primary row, with legacy fallback.

    Current D1 artifacts mark the primary rank-transfer records explicitly with
    ``analysis_type=complete_case`` and ``is_primary=True``.  Older lightweight
    fixtures lacked those fields, so keep a final fallback after preferring the
    explicit primary record.
    """
    target: list[dict[str, object]] = []
    for row in records:
        try:
            budget = int(float(str(row.get("budget_mult", ""))))
        except ValueError:
            continue
        if str(row.get("pairing_mode", "dimension")) == "dimension" and budget == 10:
            target.append(row)
    if not target:
        return records[0] if records else None

    primary_complete = [
        row for row in target
        if _truthy_value(row.get("is_primary"))
        and str(row.get("analysis_type", D1_FALLBACK_POLICY)).strip() == D1_RANK_POLICY
    ]
    if primary_complete:
        return primary_complete[0]

    primary = [row for row in target if _truthy_value(row.get("is_primary"))]
    if primary:
        return primary[0]
    return target[0]


def _select_d2_primary_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return D2 primary rows and exploratory rows, with legacy fallback."""
    if "exploratory" in df.columns:
        exp_mask = _truthy_series(df["exploratory"])
    else:
        exp_mask = pd.Series(False, index=df.index)

    main = df[~exp_mask]
    exploratory = df[exp_mask]
    if {"analysis_type", "is_primary"} <= set(df.columns):
        primary_mask = (
            df["analysis_type"].astype(str).str.strip().eq(D2_PRIMARY_ANALYSIS)
            & _truthy_series(df["is_primary"])
            & ~exp_mask
        )
        primary = df[primary_mask]
        if not primary.empty:
            return primary, exploratory
    return main, exploratory


def _load_json_if_present(path: Path) -> dict[str, object] | None:
    """Return a parsed JSON object, or ``None`` when absent/unreadable."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _collect_d3_pooled_numbers(d3_dir: Path) -> dict[str, object]:
    """Pooled D3 bias, its intervals and the robustness disclosures.

    Every D3 value quoted in the manuscript is read from a generated artifact:
    the registered pooled summary, the BCa / leave-one-case-out summary, the
    case-level sensitivity summary and the robustness-extras file produced by
    ``scripts/derive_d3_sensitivity_extras.py``.  A missing artifact yields the
    documented ``"N/A"`` sentinel, so a silent hand-typed number cannot survive
    the submission gate.
    """
    na = "N/A"
    out: dict[str, object] = {
        "dThreeNRuns": na,
        "dThreePooledBias": na,
        "dThreePooledCILo": na,
        "dThreePooledCIHi": na,
        "dThreeBCALo": na,
        "dThreeBCAHi": na,
        "dThreeLocoMin": na,
        "dThreeLocoMax": na,
        "dThreeNCases": na,
        "dThreeNCasesPositive": na,
        "dThreeCaseCILo": na,
        "dThreeCaseCIHi": na,
        "dThreeCaseP": na,
        "dThreeCaseThreshold": na,
        "dThreeNClusterExclZero": na,
        "dThreeNClusterInclZero": na,
        "dThreeNCaseHolmSurvivors": na,
        "dThreeLooCaseCILo": na,
        "dThreeLooCaseCIHi": na,
        "dThreeTopCaseSharePct": na,
        "dThreeRefScalePf": na,
        "dThreePctOfReference": na,
    }

    pooled = _load_json_if_present(d3_dir / "pooled_reliability_summary.json")
    if pooled:
        if "n_runs" in pooled:
            out["dThreeNRuns"] = int(pooled["n_runs"])
        if "pooled_mean_signed_bias" in pooled:
            out["dThreePooledBias"] = _safe_round(pooled["pooled_mean_signed_bias"], 4)
        if "cluster_bootstrap_ci_lo" in pooled:
            out["dThreePooledCILo"] = _safe_round(pooled["cluster_bootstrap_ci_lo"], 4)
        if "cluster_bootstrap_ci_hi" in pooled:
            out["dThreePooledCIHi"] = _safe_round(pooled["cluster_bootstrap_ci_hi"], 4)

    loco = _load_json_if_present(d3_dir / "d3_loco_bca.json")
    if loco:
        bca = loco.get("bca_ci") or []
        if isinstance(bca, (list, tuple)) and len(bca) == 2:
            out["dThreeBCALo"] = _safe_round(bca[0], 4)
            out["dThreeBCAHi"] = _safe_round(bca[1], 4)
        if "loco_min" in loco:
            out["dThreeLocoMin"] = _safe_round(loco["loco_min"], 4)
        if "loco_max" in loco:
            out["dThreeLocoMax"] = _safe_round(loco["loco_max"], 4)

    case_level = _load_json_if_present(d3_dir / "d3_case_level_sensitivity.json")
    if case_level:
        if "n_cases" in case_level:
            out["dThreeNCases"] = int(case_level["n_cases"])
        if "n_positive_cases" in case_level:
            out["dThreeNCasesPositive"] = int(case_level["n_positive_cases"])
        if "equal_case_bootstrap_ci_lo" in case_level:
            out["dThreeCaseCILo"] = _safe_round(case_level["equal_case_bootstrap_ci_lo"], 4)
        if "equal_case_bootstrap_ci_hi" in case_level:
            out["dThreeCaseCIHi"] = _safe_round(case_level["equal_case_bootstrap_ci_hi"], 4)
        if "wilcoxon_case_level_one_sided_greater_p" in case_level:
            out["dThreeCaseP"] = _safe_round(
                case_level["wilcoxon_case_level_one_sided_greater_p"], 4
            )
        if "holm_family_threshold" in case_level:
            out["dThreeCaseThreshold"] = _safe_round(
                case_level["holm_family_threshold"], 4
            )

    extras = _load_json_if_present(d3_dir / "d3_sensitivity_extras.json")
    if extras:
        if "n_methods_cluster_ci_excludes_zero" in extras:
            out["dThreeNClusterExclZero"] = int(
                extras["n_methods_cluster_ci_excludes_zero"]
            )
        if "n_methods_cluster_ci_includes_zero" in extras:
            out["dThreeNClusterInclZero"] = int(
                extras["n_methods_cluster_ci_includes_zero"]
            )
        if "n_methods_case_level_holm_survivors" in extras:
            out["dThreeNCaseHolmSurvivors"] = int(
                extras["n_methods_case_level_holm_survivors"]
            )
        if "loo_case_cluster_ci_lo" in extras:
            out["dThreeLooCaseCILo"] = _safe_round(extras["loo_case_cluster_ci_lo"], 4)
        if "loo_case_cluster_ci_hi" in extras:
            out["dThreeLooCaseCIHi"] = _safe_round(extras["loo_case_cluster_ci_hi"], 4)
        if "most_influential_case_share_pct_of_pooled" in extras:
            out["dThreeTopCaseSharePct"] = _safe_round(
                extras["most_influential_case_share_pct_of_pooled"], 1
            )
        if "median_true_pf_at_driving_output" in extras:
            out["dThreeRefScalePf"] = _safe_round(
                extras["median_true_pf_at_driving_output"], 2
            )
        if "pooled_bias_pct_of_median_reference_pf" in extras:
            out["dThreePctOfReference"] = int(
                round(float(extras["pooled_bias_pct_of_median_reference_pf"]))
            )
    return out


def _collect_mc_noise_numbers(test_protocol_dir: Path) -> dict[str, object]:
    """Extract test-set Monte-Carlo noise-floor macros."""
    out: dict[str, object] = {
        "mcNoiseNTest": "N/A",
        "mcNoiseMedianRelWidthPct": "N/A",
        "mcNoiseMaxRelWidthPct": "N/A",
        "mcNoiseMaxAbsWidth": "N/A",
        "mcNoiseNCases": "N/A",
    }
    csv_path = test_protocol_dir / "test_set_sufficiency.csv"
    if not csv_path.exists():
        return out

    df = pd.read_csv(csv_path)
    required = {"case_id", "n_test", "nrmse_ci_width", "nrmse_relative_ci_width"}
    if not required <= set(df.columns):
        return out
    n_test = pd.to_numeric(df["n_test"], errors="coerce")
    target = 1000 if (n_test == 1000).any() else int(n_test.dropna().max())
    sub = df[n_test == target].copy()
    if sub.empty:
        return out
    rel_width = pd.to_numeric(sub["nrmse_relative_ci_width"], errors="coerce")
    abs_width = pd.to_numeric(sub["nrmse_ci_width"], errors="coerce")
    out["mcNoiseNTest"] = int(target)
    out["mcNoiseMedianRelWidthPct"] = _safe_round(float(rel_width.median() * 100), 1)
    out["mcNoiseMaxRelWidthPct"] = _safe_round(float(rel_width.max() * 100), 1)
    out["mcNoiseMaxAbsWidth"] = _safe_round(float(abs_width.max()), 4)
    out["mcNoiseNCases"] = int(sub["case_id"].nunique())
    return out


def collect_numbers(
    results_csv: Path,
    rr_dir: Path,
    d2_dir: Path,
    d3_dir: Path,
    figdata_dir: Path,
    test_protocol_dir: Path | None = None,
) -> dict:
    nums: dict = {}

    # ── Protocol constants ───────────────────────────────────────────────
    nums["nCases"] = 34
    nums["nMethods"] = 20
    nums["nRepeats"] = 30
    nums["nBudgets"] = 4
    nums["budgetList"] = "5, 10, 20, 50"
    nums["totalCells"] = 34 * 20 * 4 * 30
    nums["seedStart"] = 1000
    nums["primaryFailurePolicy"] = "penalized_failed_in_denominator"
    nums["dOneRankPolicy"] = D1_RANK_POLICY
    nums["holmFWER"] = HOLM_FWER_ALPHA
    nums["holmFamilySize"] = HOLM_DISCOVERY_FAMILY_SIZE
    nums["holmThresholds"] = _format_thresholds(
        _holm_thresholds(HOLM_FWER_ALPHA, HOLM_DISCOVERY_FAMILY_SIZE),
    )
    nums["runDate"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    nums["gitCommit"] = _git_commit()
    if test_protocol_dir is None:
        test_protocol_dir = results_csv.parent / "analysis"

    # ── results.csv completeness ────────────────────────────────────────
    # When the sweep has not finished (or this script is run on a
    # partial-results harness) the results.csv may be absent.  We still
    # want LaTeX-safe macros, so populate every key explicitly with the
    # documented "N/A" sentinel rather than leaving "??" placeholders
    # for the tex_map fallback to fill.
    completeness_keys = (
        "totalRows", "okRows", "timeoutRows", "failedRows",
        "okFrac", "nCasesCovered", "nMethodsCovered",
    )
    if results_csv.exists():
        df = pd.read_csv(results_csv, low_memory=False)
        ok = df[df["status"] == "ok"] if "status" in df.columns else df
        nums["totalRows"] = int(len(df))
        nums["okRows"] = int((df.get("status", "ok") == "ok").sum())
        nums["timeoutRows"] = int((df.get("status", "") == "timeout").sum())
        nums["failedRows"] = int((df.get("status", "") == "failed").sum())
        nums["okFrac"] = round(len(ok) / max(len(df), 1) * 100, 1)
        nums["nCasesCovered"] = int(df["case_id"].nunique()) if "case_id" in df.columns else 0
        nums["nMethodsCovered"] = int(df["method"].nunique()) if "method" in df.columns else 0
    else:
        print(f"  [warn] {results_csv} not found; protocol counts only")
        for key in completeness_keys:
            nums[key] = "N/A"

    # ── D1 numbers ──────────────────────────────────────────────────────
    corr_json = rr_dir / "correlation.json"
    if corr_json.exists():
        corr = json.loads(corr_json.read_text(encoding="utf-8"))
        d1 = _select_d1_primary_record([
            row for row in corr
            if isinstance(row, dict)
        ])
        if d1:
            nums["dOneRankPolicy"] = str(d1.get("analysis_type", D1_FALLBACK_POLICY))
            rho = float(d1.get("spearman_rho", float("nan")))
            ci_lo = float(d1.get("spearman_rho_ci_lo", float("nan")))
            ci_hi = float(d1.get("spearman_rho_ci_hi", float("nan")))
            nums["dOneRho"] = _safe_round(rho, 3)
            nums["dOneCILo"] = _safe_round(ci_lo, 3)
            nums["dOneCIHi"] = _safe_round(ci_hi, 3)
            # The preregistration registers Kendall tau as the D1
            # sensitivity statistic alongside Spearman rho
            # ("also report Kendall tau with the ... CI"), so it is
            # exported as a first-class macro rather than left in the
            # raw correlation summary.
            nums["dOneKendallTau"] = _safe_round(
                float(d1.get("kendall_tau", float("nan"))), 3
            )
            nums["dOneKendallTauCILo"] = _safe_round(
                float(d1.get("kendall_tau_ci_lo", float("nan"))), 3
            )
            nums["dOneKendallTauCIHi"] = _safe_round(
                float(d1.get("kendall_tau_ci_hi", float("nan"))), 3
            )
            # Decision rule: a CI fully below 0.70 reads as "divergence";
            # fully above as "transfer"; straddling 0.70 (or NaN endpoints)
            # as "mixed".  Both endpoints must be finite for the strict
            # rules to fire — NaN endpoints fall through to "Mixed" by
            # design because we cannot rule out either side.
            if np.isfinite(ci_hi) and ci_hi < 0.70:
                nums["dOneDecision"] = "Divergence"
                nums["dOneVerb"] = "did not transfer"
            elif np.isfinite(ci_lo) and ci_lo > 0.70:
                nums["dOneDecision"] = "Transfer"
                nums["dOneVerb"] = "transferred reliably"
            else:
                nums["dOneDecision"] = "Mixed"
                nums["dOneVerb"] = "showed mixed evidence"
            nums["dOneNMethods"] = int(d1.get("n_methods", nums["nMethods"]))

    # The registered CI above resamples methods.  This second block adds the
    # complementary case-axis interval plus the split-half reliability of each
    # rank vector, so the manuscript can say how much of rho is attributable to
    # cross-case agreement rather than leaving that unstated.
    nums["dOneCaseAxisCILo"] = "N/A"
    nums["dOneCaseAxisCIHi"] = "N/A"
    nums["dOneCaseAxisProbBelow"] = "N/A"
    nums["dOneReliabilityAnalytical"] = "N/A"
    nums["dOneReliabilityPhysical"] = "N/A"
    nums["dOneReliabilityCeiling"] = "N/A"
    nums["dOneNPhysicalCasesComplete"] = "N/A"
    case_axis = _load_json_if_present(rr_dir / "d1_case_axis_reliability.json")
    if case_axis:
        boot = case_axis.get("case_axis_bootstrap") or {}
        if "ci_lo" in boot:
            nums["dOneCaseAxisCILo"] = _safe_round(boot["ci_lo"], 2)
        if "ci_hi" in boot:
            nums["dOneCaseAxisCIHi"] = _safe_round(boot["ci_hi"], 2)
        if "prob_below_reference" in boot:
            nums["dOneCaseAxisProbBelow"] = _safe_round(boot["prob_below_reference"], 2)
        for key, field in (
            ("dOneReliabilityAnalytical", "split_half_analytical"),
            ("dOneReliabilityPhysical", "split_half_physical"),
        ):
            block = case_axis.get(field) or {}
            if "reliability" in block:
                nums[key] = _safe_round(block["reliability"], 2)
        if "reliability_ceiling" in case_axis:
            nums["dOneReliabilityCeiling"] = _safe_round(case_axis["reliability_ceiling"], 2)
        if "n_physical_cases_complete" in case_axis:
            nums["dOneNPhysicalCasesComplete"] = int(case_axis["n_physical_cases_complete"])

    # ── D2 numbers ──────────────────────────────────────────────────────
    rs_csv = d2_dir / "rule_summary.csv"
    if rs_csv.exists():
        rs = pd.read_csv(rs_csv)
        main_rs, explo_rs = _select_d2_primary_rows(rs)
        nums["dTwoNMainTopos"] = int(len(main_rs))
        nums["dTwoNExploTopos"] = int(len(explo_rs))
        nums["dTwoNAllTopos"] = int(len(rs))
        if len(main_rs) > 0:
            frac = float((main_rs["frac_pass_at_10d"] >= 0.5).mean())
            nums["dTwoFracPassMain"] = f"{frac:.0%}"
            # Keep the practical point-estimate label separate from the
            # registered exact count-test verdict.  The former is useful for
            # descriptive reporting; the latter is the confirmatory decision
            # and must be read from the same CP artifact used by the paper.
            nums["dTwoDecision"] = "Rule Holds" if frac >= 0.5 else "Rule Fails"
            cp_path = d2_dir / "tenx_rule_cp_bound.json"
            nums["dTwoConfirmatoryDecision"] = "Inconclusive"
            if cp_path.exists():
                try:
                    cp = json.loads(cp_path.read_text(encoding="utf-8"))
                    upper = float(cp["clopper_pearson_one_sided_upper95"])
                    cutoff = float(cp.get("registered_cutoff_fraction", 4 / 7))
                    if np.isfinite(upper) and upper < cutoff:
                        nums["dTwoConfirmatoryDecision"] = "Confirmed"
                    nums["dTwoCPUpper"] = _safe_round(upper, 3)
                except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                    # Leave the conservative explicit default when the
                    # supporting artifact is absent or malformed.
                    pass
            nums["dTwoMedianNStar"] = _safe_round(
                main_rs["median_n_star"].median(), 1,
            )

    # D2 counts classes, so collapsing the three outputs with an arithmetic
    # mean rather than the worst output is a load-bearing choice.  The
    # sensitivity recomputation answers it explicitly instead of asserting it.
    nums["dTwoAltMetricPass"] = "N/A"
    nums["dTwoAltMetricNClasses"] = "N/A"
    metric_sens = _load_json_if_present(d2_dir / "d2_metric_sensitivity.json")
    if metric_sens:
        alt = metric_sens.get("alternate") or {}
        if "n_classes_point_estimate_pass" in alt:
            nums["dTwoAltMetricPass"] = int(alt["n_classes_point_estimate_pass"])
        if "n_classes" in alt:
            nums["dTwoAltMetricNClasses"] = int(alt["n_classes"])

    # ── D3 numbers ──────────────────────────────────────────────────────
    bias_csv = figdata_dir / "fig4_d3_bias.csv"
    nums.update(_collect_d3_pooled_numbers(d3_dir))

    if not bias_csv.exists():
        bias_csv = d3_dir / "reliability_per_method.csv"
    if bias_csv.exists():
        bias = pd.read_csv(bias_csv)
        rename_map = {
            "signed_bias_mean": "signed_bias",
            "signed_bias_ci_lo": "ci_lo",
            "signed_bias_ci_hi": "ci_hi",
            "signed_bias_pvalue_two_sided": "p_raw_two_sided",
            "signed_bias_pvalue_pos": "p_raw_pos",
        }
        for src, dst in rename_map.items():
            if src in bias.columns and dst not in bias.columns:
                bias[dst] = bias[src]
        if "p_holm" not in bias.columns:
            # Current submission artifacts carry the registered positive
            # one-sided p-value.  Preserve a two-sided fallback only for
            # legacy compatibility; the strict submission gate rejects a
            # table that lacks the positive column.
            p_col = "p_raw_pos" if "p_raw_pos" in bias.columns else "p_raw_two_sided"
            if p_col in bias.columns:
                bias["p_holm"] = _holm_adjust(bias[p_col].to_numpy(dtype=float))
        if "p_holm" not in bias.columns:
            print(
                "  [warn] D3 bias table lacks p_holm or p_raw_two_sided; "
                "leaving D3 headline numbers as N/A",
            )
            nums.update({
                "dThreeNSigMethods": "N/A",
                "dThreeNTested": "N/A",
                "dThreeNLowN": "N/A",
                "dThreeDecision": "N/A",
                "dThreeDirection": "N/A",
            })
            nums.update(_collect_mc_noise_numbers(test_protocol_dir))
            return nums
        if "low_n" not in bias.columns:
            if "n_runs" in bias.columns:
                n_runs = pd.to_numeric(bias["n_runs"], errors="coerce")
                bias["low_n"] = n_runs < D3_MIN_RUNS_PER_METHOD
            else:
                bias["low_n"] = False
        low_n = _truthy_series(bias["low_n"])
        p_holm = bias.get("p_holm", pd.Series([np.inf] * len(bias), index=bias.index))
        primary = bias[~low_n]
        sig = primary[p_holm.loc[primary.index] < 0.05]
        nums["dThreeNSigMethods"] = int(len(sig))
        nums["dThreeNTested"] = int(len(primary))
        nums["dThreeNLowN"] = int(low_n.sum())
        nums["dThreeDecision"] = "Bias Exists" if len(sig) > 0 else "No Bias"
        if len(sig) > 0 and "signed_bias" in sig.columns:
            nums["dThreeDirection"] = _classify_direction(sig["signed_bias"])
        else:
            nums["dThreeDirection"] = "none"

    nums.update(_collect_mc_noise_numbers(test_protocol_dir))

    return nums


def _latex_escape_value(val: object) -> str:
    """Escape LaTeX-special characters in a macro replacement value.

    Generated values are raw (never pre-escaped), so a literal ``%`` would
    comment out the rest of ``numbers.tex`` and underscores in policy names
    (e.g. ``penalized_failed_in_denominator``) would error in text mode.
    Only characters that actually occur in these numeric/word values are
    escaped; ``\\``, ``{`` and ``}`` are deliberately left untouched.
    """
    text = str(val)
    for ch in ("&", "%", "#", "_"):
        text = text.replace(ch, "\\" + ch)
    return text


def write_tex(nums: dict, path: Path, *, verbose: bool = True) -> None:
    lines = [
        "% paper/numbers.tex — auto-generated by fill_paper_numbers.py",
        f"% Generated: {nums.get('runDate', '?')}  Commit: {nums.get('gitCommit', '?')}",
        "% DO NOT EDIT MANUALLY",
        "",
    ]
    tex_map = {
        "nCases": r"\nCases",
        "nMethods": r"\nMethods",
        "nRepeats": r"\nRepeats",
        "nBudgets": r"\nBudgets",
        "budgetList": r"\budgetList",
        "totalCells": r"\totalCells",
        "seedStart": r"\seedStart",
        "primaryFailurePolicy": r"\primaryFailurePolicy",
        "dOneRankPolicy": r"\dOneRankPolicy",
        "holmFWER": r"\HolmFWER",
        "holmFamilySize": r"\HolmFamilySize",
        "holmThresholds": r"\HolmThresholds",
        "mcNoiseNTest": r"\MCNoiseNTest",
        "mcNoiseMedianRelWidthPct": r"\MCNoiseMedianRelWidthPct",
        "mcNoiseMaxRelWidthPct": r"\MCNoiseMaxRelWidthPct",
        "mcNoiseMaxAbsWidth": r"\MCNoiseMaxAbsWidth",
        "mcNoiseNCases": r"\MCNoiseNCases",
        "okFrac": r"\okFrac",
        "runDate": r"\runDate",
        "gitCommit": r"\gitCommit",
        "dOneRho": r"\dOneRho",
        "dOneCILo": r"\dOneCILo",
        "dOneCIHi": r"\dOneCIHi",
        "dOneKendallTau": r"\dOneKendallTau",
        "dOneKendallTauCILo": r"\dOneKendallTauCILo",
        "dOneKendallTauCIHi": r"\dOneKendallTauCIHi",
        "dOneDecision": r"\dOneDecision",
        "dOneVerb": r"\dOneVerb",
        "dOneNMethods": r"\dOneNMethods",
        "dOneCaseAxisCILo": r"\dOneCaseAxisCILo",
        "dOneCaseAxisCIHi": r"\dOneCaseAxisCIHi",
        "dOneCaseAxisProbBelow": r"\dOneCaseAxisProbBelow",
        "dOneReliabilityAnalytical": r"\dOneReliabilityAnalytical",
        "dOneReliabilityPhysical": r"\dOneReliabilityPhysical",
        "dOneReliabilityCeiling": r"\dOneReliabilityCeiling",
        "dOneNPhysicalCasesComplete": r"\dOneNPhysicalCasesComplete",
        "dTwoNMainTopos": r"\dTwoNMainTopos",
        "dTwoNExploTopos": r"\dTwoNExploTopos",
        "dTwoFracPassMain": r"\dTwoFracPassMain",
        "dTwoDecision": r"\dTwoDecision",
        "dTwoConfirmatoryDecision": r"\dTwoConfirmatoryDecision",
        "dTwoMedianNStar": r"\dTwoMedianNStar",
        "dTwoCPUpper": r"\dTwoCPUpper",
        "dTwoAltMetricPass": r"\dTwoAltMetricPass",
        "dTwoAltMetricNClasses": r"\dTwoAltMetricNClasses",
        "dThreeNSigMethods": r"\dThreeNSigMethods",
        "dThreeNTested": r"\dThreeNTested",
        "dThreeNLowN": r"\dThreeNLowN",
        "dThreeDecision": r"\dThreeDecision",
        "dThreeDirection": r"\dThreeDirection",
        "dThreeNRuns": r"\dThreeNRuns",
        "dThreePooledBias": r"\dThreePooledBias",
        "dThreePooledCILo": r"\dThreePooledCILo",
        "dThreePooledCIHi": r"\dThreePooledCIHi",
        "dThreeBCALo": r"\dThreeBCALo",
        "dThreeBCAHi": r"\dThreeBCAHi",
        "dThreeLocoMin": r"\dThreeLocoMin",
        "dThreeLocoMax": r"\dThreeLocoMax",
        "dThreeNCases": r"\dThreeNCases",
        "dThreeNCasesPositive": r"\dThreeNCasesPositive",
        "dThreeCaseCILo": r"\dThreeCaseCILo",
        "dThreeCaseCIHi": r"\dThreeCaseCIHi",
        "dThreeCaseP": r"\dThreeCaseP",
        "dThreeCaseThreshold": r"\dThreeCaseThreshold",
        "dThreeNClusterExclZero": r"\dThreeNClusterExclZero",
        "dThreeNClusterInclZero": r"\dThreeNClusterInclZero",
        "dThreeNCaseHolmSurvivors": r"\dThreeNCaseHolmSurvivors",
        "dThreeLooCaseCILo": r"\dThreeLooCaseCILo",
        "dThreeLooCaseCIHi": r"\dThreeLooCaseCIHi",
        "dThreeTopCaseSharePct": r"\dThreeTopCaseSharePct",
        "dThreeRefScalePf": r"\dThreeRefScalePf",
        "dThreePctOfReference": r"\dThreePctOfReference",
    }
    for key, macro in tex_map.items():
        value = nums.get(key, "??")
        if key == "totalCells" and isinstance(value, int):
            # Thousands separator, to match how the SI and the reporting
            # summary quote the same cell count (81,600).  Only the typeset
            # form changes; numbers.json keeps the integer, which the
            # submission gate compares against the manifest.
            value = f"{value:,}"
        val = _latex_escape_value(value)
        lines.append(rf"\newcommand{{{macro}}}{{{val}}}")
        num_macro = _num_macro_name(key)
        lines.append(rf"\newcommand{{{num_macro}}}{{{val}}}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if verbose:
        print(f"  [ok] {path}: {len(tex_map) * 2} macros written")


def _ts_slug() -> str:
    return datetime.now(timezone.utc).isoformat().replace(":", "").replace("+", "Z")


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _replace_numbers_package_atomically(
    src_tex: Path,
    dst_tex: Path,
    src_json: Path,
    dst_json: Path,
) -> None:
    """Replace numbers.tex and numbers.json as a consistent pair."""
    src_tex = src_tex.resolve()
    dst_tex = dst_tex.resolve()
    src_json = src_json.resolve()
    dst_json = dst_json.resolve()
    dst_tex.parent.mkdir(parents=True, exist_ok=True)
    dst_json.parent.mkdir(parents=True, exist_ok=True)

    suffix = _ts_slug()
    backups: dict[Path, Path] = {}
    for dst in (dst_tex, dst_json):
        if not dst.exists():
            continue
        backup = dst.with_name(f".{dst.name}.backup.{suffix}")
        _remove_path(backup)
        dst.rename(backup)
        backups[dst] = backup

    try:
        shutil.move(str(src_tex), str(dst_tex))
        shutil.move(str(src_json), str(dst_json))
    except Exception:
        _remove_path(dst_tex)
        _remove_path(dst_json)
        for dst, backup in backups.items():
            if backup.exists():
                backup.rename(dst)
        raise

    for backup in backups.values():
        _remove_path(backup)


def write_numbers_package(nums: dict, output_tex: Path, output_json: Path) -> None:
    """Write paper-number artefacts only after both files are complete."""
    tmp_parent_base = output_tex.resolve().parent
    tmp_parent_base.mkdir(parents=True, exist_ok=True)
    tmp_parent = Path(tempfile.mkdtemp(
        prefix=f".{output_tex.stem}.tmp.",
        dir=str(tmp_parent_base),
    ))
    tmp_tex = tmp_parent / output_tex.name
    tmp_json = tmp_parent / output_json.name
    try:
        write_tex(nums, tmp_tex, verbose=False)
        tmp_json.write_text(
            json.dumps(nums, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _replace_numbers_package_atomically(
            tmp_tex,
            output_tex,
            tmp_json,
            output_json,
        )
    finally:
        if tmp_parent.exists():
            shutil.rmtree(tmp_parent)


def _num_macro_name(key: str) -> str:
    if key.startswith("mcNoise"):
        return rf"\NumMCNoise{key[len('mcNoise'):]}"
    if key.startswith("n") and len(key) > 1 and key[1].isupper():
        return rf"\NumN{key[1:]}"
    return rf"\Num{key[0].upper()}{key[1:]}"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute and emit all numerical macros (D1/D2/D3 + protocol) "
            "for manuscript numbers.tex + numbers.json (GAP-8)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--results-csv", type=Path,
        default=ROOT / "benchmark" / "results" / "full_benchmark" / "results.csv",
    )
    p.add_argument(
        "--rr-dir", type=Path,
        default=ROOT / "benchmark" / "results" / "ranking_reversal",
    )
    p.add_argument(
        "--d2-dir", type=Path,
        default=ROOT / "benchmark" / "results" / "sample_complexity_0.20",
    )
    p.add_argument(
        "--d3-dir", type=Path,
        default=ROOT / "benchmark" / "results" / "reliability_recovery",
    )
    p.add_argument(
        "--figdata-dir", type=Path,
        default=FIGDATA_DIR,
    )
    p.add_argument(
        "--test-protocol-dir", type=Path,
        default=None,
        help=(
            "Directory containing test_set_sufficiency.csv from "
            "benchmark/scripts/validate_test_protocol.py. Defaults to "
            "<results-csv parent>/analysis."
        ),
    )
    p.add_argument(
        "--output-tex", type=Path,
        default=MANUSCRIPTS_DIR / "numbers.tex",
    )
    p.add_argument(
        "--output-json", type=Path,
        default=MANUSCRIPTS_DIR / "numbers.json",
    )
    p.add_argument(
        "--require-results-csv",
        action="store_true",
        help=(
            "Fail instead of writing N/A completeness macros when results.csv "
            "is missing. Use this in the formal paper pipeline."
        ),
    )
    p.add_argument(
        "--require-results-contract",
        action="store_true",
        help=(
            "Fail before writing paper-number artifacts unless results.csv "
            "passes the same corruption checks used by prepare_figdata.py."
        ),
    )
    p.add_argument(
        "--require-headline-source",
        action="store_true",
        help=(
            "Fail unless --results-csv is the canonical full_benchmark/results.csv "
            "and the CSV is the exact frozen 34 x 20 x 4 x 30 nature-pool grid. "
            "Use this in the formal paper pipeline."
        ),
    )
    p.add_argument(
        "--require-submission-complete",
        action="store_true",
        help=(
            "Fail instead of writing manuscript numbers outputs when any "
            "submission-critical macro would still be N/A."
        ),
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.require_results_csv and not args.results_csv.exists():
        print(f"[ERROR] required results.csv not found: {args.results_csv}", file=sys.stderr)
        return 2
    if args.require_results_contract:
        if not args.results_csv.exists():
            print(f"[ERROR] required results.csv not found: {args.results_csv}", file=sys.stderr)
            return 2
        try:
            from scripts.prepare_figdata import validate_results_contract

            validate_results_contract(pd.read_csv(args.results_csv, low_memory=False))
        except AssertionError as exc:
            print(f"[ERROR] results.csv contract failed: {exc}", file=sys.stderr)
            return 2
    if args.require_headline_source:
        if not args.results_csv.exists():
            print(f"[ERROR] required results.csv not found: {args.results_csv}", file=sys.stderr)
            return 2
        try:
            from scripts.prepare_figdata import (
                validate_headline_results_source,
                validate_results_contract,
            )

            df = pd.read_csv(args.results_csv, low_memory=False)
            validate_results_contract(df)
            validate_headline_results_source(args.results_csv, df)
        except AssertionError as exc:
            print(f"[ERROR] headline results contract failed: {exc}", file=sys.stderr)
            return 2
    print("Collecting paper numbers...")
    nums = collect_numbers(
        args.results_csv, args.rr_dir, args.d2_dir, args.d3_dir, args.figdata_dir,
        args.test_protocol_dir,
    )
    nums["_provenance"] = numbers_provenance(
        args.results_csv,
        args.rr_dir,
        args.d2_dir,
        args.d3_dir,
        args.figdata_dir,
        args.test_protocol_dir,
    )
    if args.require_submission_complete:
        errors = primary_policy_input_errors(args.rr_dir, args.d2_dir)
        errors.extend(submission_completeness_errors(nums))
        if errors:
            for err in errors:
                print(f"[ERROR] {err}", file=sys.stderr)
            return 3
    write_numbers_package(nums, args.output_tex, args.output_json)
    print(f"  [ok] {args.output_tex}: paper-number macros written")
    print(f"  [ok] {args.output_json}: {len(nums)} numbers exported")
    print("\nKey numbers:")
    for k in [
        "dOneRho", "dOneCILo", "dOneCIHi", "dOneDecision",
        "dTwoFracPassMain", "dTwoDecision",
        "dThreeNSigMethods", "dThreeDecision",
    ]:
        print(f"  {k} = {nums.get(k, 'N/A')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
