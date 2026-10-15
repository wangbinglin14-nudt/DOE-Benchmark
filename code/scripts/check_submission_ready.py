"""check_submission_ready.py
============================

Fail-fast CI guard for paper submission readiness.  It confirms no unresolved
manuscript placeholders remain in submission sources and verifies that the run
manifest's archived artifacts still exist with matching size/hash.

Usage
-----

    python scripts/check_submission_ready.py [paper_dir]
    python scripts/check_submission_ready.py ../manuscripts/

The default command uses the formal full-archive checks, with manuscript
artifacts under the repository-root ``manuscripts/`` directory: it checks
``benchmark/results/full_benchmark/RUN_MANIFEST.json``, ``numbers.*``,
``figure_manifest.json``, ``figdata/``, D1/D2/D3 analysis
directories, the full-sweep test-protocol diagnostics, and the case-validation
ledgers unless the caller explicitly passes a ``--skip-*`` flag.
It also requires the full manuscript sources; the public artifact-only
``manuscripts/`` directory is not a complete submission package.

Return codes
------------

* ``0``: no manuscript placeholders, manifest artifacts are present, and
  paper numbers/figures are hash-clean against their manifests.
* ``1``: one or more readiness checks failed; every hit is printed with
  actionable detail.

Design rules
------------

* Scan ``*.tex``, ``*.md``, ``*.bib`` under the target directory.
* **Skip the tracking documents themselves** — ``SUBMISSION_CHECKLIST.md``
  and ``REVIEWER_CRITIQUE.md`` intentionally quote the string to remind
  the author what still needs filling in, and failing on them would
  defeat the purpose of the checklist.
* Also skip files inside a ``_build/``, ``aux/``, or other LaTeX-output
  directory that may be produced by a local compile.
* Template files such as ``*.template.md`` are ignored; they intentionally
  contain bracketed fields and are not submitted directly.
* Unresolved tokens are treated case-sensitively and include all-caps
  placeholders such as ``PLACEHOLDER``, ``TBD``, ``TODO``, ``XXX``, and
  bracketed result stubs such as ``[X.XXX]`` or ``[L, H]``.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import io
import json
import math
import re
import shlex
import sys
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable

# Lines of the form ``\newcommand{\NumXxx}{PLACEHOLDER}`` are
# intentional stub defaults used by the manuscripts' ``\IfFileExists``
# fallback block; they only render if ``numbers.tex`` is missing at
# compile time.  Once ``numbers.tex`` is generated the real values
# override them, so they are dead code at submission time and should
# not cause this guard to fail.
_STUB_FALLBACK_RE = re.compile(
    r"^\s*\\newcommand\{\\Num[A-Za-z0-9]+\}\{PLACEHOLDER\}\s*%?.*$"
)
_UNRESOLVED_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("PLACEHOLDER", re.compile(r"\bPLACEHOLDER\b")),
    ("TBD", re.compile(r"\bTBD\b")),
    ("TODO", re.compile(r"\bTODO\b")),
    ("XXX", re.compile(r"\bXXX\b")),
    ("RESULT_PENDING", re.compile(r"\bRESULT(?:\\_|\_)?PENDING\b")),
    ("DOI_TBD", re.compile(r"\bDOI\s*:\s*TBD\b")),
    ("BRACKETED_RESULT", re.compile(r"\[(?:X(?:\.X+)?|L,\s*H|p(?:_holm)?|Y/N|Yes\s*\|\s*No|N/A)\]")),
)
_PIP_FREEZE_VERSION_RE = re.compile(r"^([A-Za-z0-9_.-]+)==(.+)$")
_DIST_NAME_NORMALIZE_RE = re.compile(r"[-_.]+")
TEMP_STORAGE_ENV_VARS = (
    "TMPDIR",
    "TMP",
    "TEMP",
    "TEMPDIR",
    "SLURM_TMPDIR",
    "SCRATCH",
    "LOCAL_SCRATCH",
    "JOBSCRATCH",
)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.project_layout import MANUSCRIPTS_DIR, paper_file, metadata_file
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.runtime_contracts import (  # noqa: E402
    ALLOWED_FAILURE_CATEGORIES,
    invalid_runtime_duration_mask,
)

PAPER_DIR_DEFAULT = MANUSCRIPTS_DIR
MANIFEST_DEFAULT = ROOT / "benchmark" / "results" / "full_benchmark" / "RUN_MANIFEST.json"
NUMBERS_DEFAULT = PAPER_DIR_DEFAULT / "numbers.json"
NUMBERS_TEX_DEFAULT = PAPER_DIR_DEFAULT / "numbers.tex"
FIGURE_MANIFEST_DEFAULT = PAPER_DIR_DEFAULT / "figure_manifest.json"
D1_DIR_DEFAULT = ROOT / "benchmark" / "results" / "ranking_reversal"
D2_DIR_DEFAULT = ROOT / "benchmark" / "results" / "sample_complexity_0.20"
D3_DIR_DEFAULT = ROOT / "benchmark" / "results" / "reliability_recovery"
TEST_PROTOCOL_DIR_DEFAULT = ROOT / "benchmark" / "results" / "full_benchmark" / "analysis"
PRIMARY_D1_ANALYSIS = "complete_case"
PRIMARY_FIG6_ANALYSIS = "penalized_failed_in_denominator"
PRIMARY_D2_ANALYSIS = "penalized_failed_in_denominator"
PRIMARY_D3_CLUSTER_BY = "case_id"
D3_SENSITIVITY_CLUSTER_BY = "iid"
EXPECTED_FIGDATA_ANALYSIS_POLICY = {
    "schema_version": 1,
    "d1_primary_analysis": PRIMARY_D1_ANALYSIS,
    "d2_primary_analysis": PRIMARY_D2_ANALYSIS,
    "fig6_primary_analysis": PRIMARY_FIG6_ANALYSIS,
    "d3_primary_cluster_by": PRIMARY_D3_CLUSTER_BY,
    "d3_sensitivity_cluster_by": D3_SENSITIVITY_CLUSTER_BY,
}
PREREG_POOL = "nature"
PREREG_BUDGETS = [5, 10, 20, 50]
PREREG_N_REPEATS = 30
PREREG_N_TEST = 1000
PREREG_SEED_STRATEGY = "case_repeat"
PREREG_SEED_START = 1000
PREREG_N_CASES = 34
PREREG_N_METHODS = 20
CASE_VALIDATION_STATUS_DEFAULT = ROOT / "docs" / "case_validation_status.csv"
CASE_VALIDATION_SOURCES_DEFAULT = ROOT / "docs" / "case_validation_sources.csv"
CASE_VALIDATION_NOTE_SOURCES_DEFAULT = ROOT / "docs" / "case_validation_note_sources.csv"
CASE_VALIDATION_SUMMARY_DEFAULT = ROOT / "docs" / "CASE_VALIDATION_SUMMARY.md"
CASE_EXPERIMENT_READINESS_DEFAULT = ROOT / "docs" / "case_experiment_readiness.csv"
CASE_VALIDATION_REQUIRED_COLUMNS = [
    "case_id",
    "domain",
    "dim",
    "validation_note_path",
    "external_validation_scope",
    "formula_status",
    "code_status",
    "boundary_test_status",
    "targeted_physics_test",
    "risk_level",
    "final_disposition",
    "next_action",
]
CASE_VALIDATION_FORMULA_STATUS = {
    "fixed_targeted_tested",
    "targeted_tested_no_code_change",
    "cross_validation_only",
}
CASE_VALIDATION_ALLOWED_RISKS = {"P1", "P2"}
CASE_VALIDATION_SOURCE_COLUMNS = [
    "case_id",
    "case_source_path",
    "case_source_sha256",
]
CASE_VALIDATION_NOTE_SOURCE_COLUMNS = [
    "path",
    "size_bytes",
    "sha256",
]
REQUIRED_SUBMISSION_SOURCES = (
    "nature_cs_main.tex",
    "preregistration_decisions.md",
    "significance_statement.md",
    "cover_letter.md",
)
ALLOWED_DISCOVERY_DECISIONS = {"Confirmed", "Falsified", "Inconclusive"}
ALLOWED_BINARY_DECISIONS = {"Y", "N", "Yes", "No", "True", "False"}
SIGNIFICANCE_WORD_MIN = 100
SIGNIFICANCE_WORD_MAX = 200
COVER_LETTER_WORD_MAX = 450
COVER_LETTER_REQUIRED_PATTERNS = (
    (r"\bDear\s+Editor\b", "editor salutation"),
    (r"\bYours\s+sincerely\b", "closing signature line"),
    (r"DOE-Benchmark", "manuscript title"),
    (r"physics-grounded", "physics-grounded positioning"),
    (r"20\s+design-of-experiments", "20-method benchmark scope"),
    (r"34\s+physics-grounded", "34-case benchmark scope"),
    # Provenance/hashes are validated from the actual manifest below; internal
    # filenames and checksums are not editorial cover-letter requirements.
    (r"(?:immutable|versioned\s+delivery)\s+archive", "archive/data availability plan"),
)
# The manuscript deliberately avoids the internal `P1`/`P2` risk-level labels:
# `P2` in docs/case_validation_status.csv is a risk_level value, not an evidence
# tier, and five of the P2 cases are analytical controls rather than
# plausibility-only.  The gate therefore checks the neutral vocabulary that the
# manuscript actually uses.
CASE_VALIDATION_CAVEAT_PATTERNS = (
    re.compile(r"evidence\s+tiers", re.IGNORECASE),
    re.compile(r"plausibility-only", re.IGNORECASE),
    re.compile(
        r"(?:no|not)\s+strong\s+external\s+case-level\s+validation\s+claims",
        re.IGNORECASE,
    ),
)
REQUIRED_SI_FILES = (
    "si_hyperparameters.tex",
    "si_hyperparameters.md",
    "si_sobol.tex",
    "si_sobol.md",
    "si_cases.tex",
    "si_cases.md",
    "si_multiple_testing.tex",
    "si_multiple_testing.md",
    "si_d2_policy.tex",
    "si_d2_policy.md",
)
REQUIRED_CASE_EVIDENCE_TIERS = {
    "external_note_targeted_regression",
    "external_note_analytic_control",
    "targeted_analytical_control",
    "targeted_plausibility_only",
}
MIN_P2_PLAUSIBILITY_TEST_FUNCTIONS = 3
MIN_P2_PLAUSIBILITY_PHYSICS_TEST_FUNCTIONS = 2
GENERIC_CASE_PHYSICS_TEST_NAME_FRAGMENTS = (
    "declared_output_ranges",
    "backend_is_used",
)

# File names that are *expected* to mention PLACEHOLDER and must be
# skipped even if they are under ``paper/``.
SKIP_BASENAMES: set[str] = {
    "SUBMISSION_CHECKLIST.md",
    "REVIEWER_CRITIQUE.md",
}

# Directory components that are always skipped (LaTeX build output).
SKIP_DIRS: set[str] = {
    "_build", "aux", ".git", ".venv", "build", "dist",
}

EXTENSIONS: tuple[str, ...] = (".tex", ".md", ".bib")
NUMBER_MACROS: dict[str, tuple[str, str]] = {
    "nCases": (r"\nCases", r"\NumNCases"),
    "nMethods": (r"\nMethods", r"\NumNMethods"),
    "nRepeats": (r"\nRepeats", r"\NumNRepeats"),
    "nBudgets": (r"\nBudgets", r"\NumNBudgets"),
    "budgetList": (r"\budgetList", r"\NumBudgetList"),
    "totalCells": (r"\totalCells", r"\NumTotalCells"),
    "seedStart": (r"\seedStart", r"\NumSeedStart"),
    "primaryFailurePolicy": (r"\primaryFailurePolicy", r"\NumPrimaryFailurePolicy"),
    "dOneRankPolicy": (r"\dOneRankPolicy", r"\NumDOneRankPolicy"),
    "holmFWER": (r"\HolmFWER", r"\NumHolmFWER"),
    "holmFamilySize": (r"\HolmFamilySize", r"\NumHolmFamilySize"),
    "holmThresholds": (r"\HolmThresholds", r"\NumHolmThresholds"),
    "mcNoiseNTest": (r"\MCNoiseNTest", r"\NumMCNoiseNTest"),
    "mcNoiseMedianRelWidthPct": (r"\MCNoiseMedianRelWidthPct", r"\NumMCNoiseMedianRelWidthPct"),
    "mcNoiseMaxRelWidthPct": (r"\MCNoiseMaxRelWidthPct", r"\NumMCNoiseMaxRelWidthPct"),
    "mcNoiseMaxAbsWidth": (r"\MCNoiseMaxAbsWidth", r"\NumMCNoiseMaxAbsWidth"),
    "mcNoiseNCases": (r"\MCNoiseNCases", r"\NumMCNoiseNCases"),
    "okFrac": (r"\okFrac", r"\NumOkFrac"),
    "runDate": (r"\runDate", r"\NumRunDate"),
    "gitCommit": (r"\gitCommit", r"\NumGitCommit"),
    "dOneRho": (r"\dOneRho", r"\NumDOneRho"),
    "dOneCILo": (r"\dOneCILo", r"\NumDOneCILo"),
    "dOneCIHi": (r"\dOneCIHi", r"\NumDOneCIHi"),
    "dOneDecision": (r"\dOneDecision", r"\NumDOneDecision"),
    "dOneVerb": (r"\dOneVerb", r"\NumDOneVerb"),
    "dOneNMethods": (r"\dOneNMethods", r"\NumDOneNMethods"),
    "dTwoNMainTopos": (r"\dTwoNMainTopos", r"\NumDTwoNMainTopos"),
    "dTwoNExploTopos": (r"\dTwoNExploTopos", r"\NumDTwoNExploTopos"),
    "dTwoFracPassMain": (r"\dTwoFracPassMain", r"\NumDTwoFracPassMain"),
    "dTwoDecision": (r"\dTwoDecision", r"\NumDTwoDecision"),
    "dTwoConfirmatoryDecision": (r"\dTwoConfirmatoryDecision", r"\NumDTwoConfirmatoryDecision"),
    "dTwoMedianNStar": (r"\dTwoMedianNStar", r"\NumDTwoMedianNStar"),
    "dThreeNSigMethods": (r"\dThreeNSigMethods", r"\NumDThreeNSigMethods"),
    "dThreeNTested": (r"\dThreeNTested", r"\NumDThreeNTested"),
    "dThreeNLowN": (r"\dThreeNLowN", r"\NumDThreeNLowN"),
    "dThreeDecision": (r"\dThreeDecision", r"\NumDThreeDecision"),
    "dThreeDirection": (r"\dThreeDirection", r"\NumDThreeDirection"),
}
NUMBERS_SUBMISSION_REQUIRED_NON_NA = {
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
    "dOneDecision",
    "dOneVerb",
    "dOneNMethods",
    "dTwoNMainTopos",
    "dTwoNExploTopos",
    "dTwoFracPassMain",
    "dTwoDecision",
    "dTwoMedianNStar",
    "dThreeNSigMethods",
    "dThreeNTested",
    "dThreeNLowN",
    "dThreeDecision",
    "dThreeDirection",
    "mcNoiseNTest",
    "mcNoiseMedianRelWidthPct",
    "mcNoiseMaxRelWidthPct",
    "mcNoiseMaxAbsWidth",
    "mcNoiseNCases",
}

NUMBERS_PROVENANCE_REQUIRED_LABELS = {
    "results_csv",
    "d1_correlation",
    "d2_rule_summary",
    "d3_bias_numbers",
    "test_set_sufficiency",
}

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_NEWCOMMAND_RE = re.compile(r"\\newcommand\{\\([A-Za-z][A-Za-z0-9]*)\}\{([^{}]*)\}")
FIGDATA_PROVENANCE_COLUMNS = {"_source", "_generated_at"}
FIGDATA_SCHEMAS: dict[str, dict[str, object]] = {
    "fig1_case_catalog.csv": {
        "required": {"case_id", "domain", "dim", "n_output", "topology", "has_failure", "supports_mf"},
        "numeric": {"dim", "n_output"},
        "boolean": {"has_failure", "supports_mf"},
    },
    "fig1_method_catalog.csv": {
        "required": {"method", "family", "surrogate", "pool"},
    },
    "fig2_d1_spearman.csv": {
        "required": {"pairing_mode", "budget_mult", "spearman_rho", "spearman_p", "kendall_tau", "analysis_type", "is_primary"},
        "numeric": {"budget_mult", "spearman_rho", "spearman_p", "kendall_tau"},
        "boolean": {"is_primary"},
    },
    "fig2_d1_ranks.csv": {
        "required": {"pairing_mode", "budget_mult", "method", "rank_analytical", "rank_physical", "analysis_type", "is_primary"},
        "numeric": {"budget_mult", "rank_analytical", "rank_physical"},
        "boolean": {"is_primary"},
    },
    "fig3_d2_learning.csv": {
        "required": {"topology", "method", "budget_mult", "median", "ci_lo", "ci_hi", "n_cases"},
        "numeric": {"budget_mult", "median", "ci_lo", "ci_hi", "n_cases"},
    },
    "fig3_d2_rule.csv": {
        "required": {"topology", "frac_pass_at_10d", "median_n_star", "n_cases", "exploratory", "analysis_type", "is_primary"},
        "numeric": {"frac_pass_at_10d", "n_cases"},
        "boolean": {"exploratory", "is_primary"},
    },
    "fig4_d3_bias.csv": {
        "required": {"method", "signed_bias", "ci_lo", "ci_hi", "p_holm", "bias_direction", "n_runs", "low_n"},
        "numeric": {"signed_bias", "ci_lo", "ci_hi", "p_holm", "n_runs"},
        "boolean": {"low_n"},
    },
    "fig5_mechanism.csv": {
        "required": {"topology", "method", "budget_mult", "n_cases"},
        "one_of": [{"mean_rank", "rank", "nrmse_rank"}],
        "numeric": {"budget_mult", "n_cases"},
    },
    "fig6_pareto.csv": {
        "required": {"method", "nrmse_median", "budget_mult", "analysis_type", "on_pareto", "is_primary"},
        "one_of": [{"time_median_s", "wall_time_median"}],
        "numeric": {"nrmse_median", "budget_mult", "time_median_s", "wall_time_median"},
        "boolean": {"on_pareto", "is_primary"},
    },
}
ANALYSIS_ARTIFACT_SCHEMAS: dict[str, dict[str, object]] = {
    "d1_correlation_json": {
        "required": {
            "pairing_mode",
            "budget_mult",
            "spearman_rho",
            "spearman_rho_ci_lo",
            "spearman_rho_ci_hi",
            "kendall_tau",
            "kendall_tau_ci_lo",
            "kendall_tau_ci_hi",
            "analysis_type",
            "is_primary",
        },
        "numeric": {
            "budget_mult",
            "spearman_rho",
            "spearman_rho_ci_lo",
            "spearman_rho_ci_hi",
            "kendall_tau",
            "kendall_tau_ci_lo",
            "kendall_tau_ci_hi",
        },
        "expected_values": {
            "analysis_type": {PRIMARY_D1_ANALYSIS},
        },
        "required_pairing_modes": {"dimension", "topology", "random"},
        "required_budgets": {10, 20},
        "required_primary_pairing_modes": {"dimension", "topology", "random"},
        "required_primary_budgets": {10, 20},
    },
    "d1_rank_comparison.csv": {
        "required": {
            "pairing_mode",
            "budget_mult",
            "method",
            "rank_analytical",
            "rank_physical",
            "rank_shift",
            "analysis_type",
            "is_primary",
        },
        "numeric": {"budget_mult", "rank_analytical", "rank_physical", "rank_shift"},
        "expected_values": {
            "analysis_type": {PRIMARY_D1_ANALYSIS},
        },
        "required_budgets": {10, 20},
        "required_primary_budgets": {10, 20},
    },
    "learning_curves.csv": {
        "required": {
            "topology",
            "method",
            "budget_mult",
            "median",
            "ci_lo",
            "ci_hi",
            "n_cases",
            "n_runs",
        },
        "numeric": {"budget_mult", "median", "ci_lo", "ci_hi", "n_cases", "n_runs"},
    },
    "rule_summary.csv": {
        "required": {
            "topology",
            "analysis_type",
            "is_primary",
            "methods_total",
            "methods_pass_at_10d",
            "frac_pass_at_10d",
            "median_n_star",
            "max_n_star",
            "n_methods_never_pass",
            "n_cases",
            "case_ids",
            "exploratory",
            "n_failed_cells_per_topology",
        },
        "numeric": {
            "methods_total",
            "methods_pass_at_10d",
            "frac_pass_at_10d",
            "n_methods_never_pass",
            "n_cases",
            "n_failed_cells_per_topology",
        },
        "expected_values": {
            "analysis_type": {PRIMARY_D2_ANALYSIS},
        },
    },
    "reliability_per_method.csv": {
        "required": {
            "method",
            "method_family",
            "n_runs",
            "low_n",
            "pf_abs_err_mean",
            "pf_abs_err_ci_lo",
            "pf_abs_err_ci_hi",
            "signed_bias_mean",
            "signed_bias_ci_lo",
            "signed_bias_ci_hi",
            "signed_bias_pvalue_two_sided",
            "signed_bias_pvalue_pos",
            "p_holm",
        },
        "numeric": {
            "n_runs",
            "pf_abs_err_mean",
            "pf_abs_err_ci_lo",
            "pf_abs_err_ci_hi",
            "signed_bias_mean",
            "signed_bias_ci_lo",
            "signed_bias_ci_hi",
            "signed_bias_pvalue_two_sided",
            "signed_bias_pvalue_pos",
            "p_holm",
        },
        "check_low_n": True,
    },
    "reliability_per_method_iid.csv": {
        "required": {
            "method",
            "method_family",
            "n_runs",
            "low_n",
            "pf_abs_err_mean",
            "pf_abs_err_ci_lo",
            "pf_abs_err_ci_hi",
            "signed_bias_mean",
            "signed_bias_ci_lo",
            "signed_bias_ci_hi",
            "signed_bias_pvalue_two_sided",
            "signed_bias_pvalue_pos",
            "p_holm",
        },
        "numeric": {
            "n_runs",
            "pf_abs_err_mean",
            "pf_abs_err_ci_lo",
            "pf_abs_err_ci_hi",
            "signed_bias_mean",
            "signed_bias_ci_lo",
            "signed_bias_ci_hi",
            "signed_bias_pvalue_two_sided",
            "signed_bias_pvalue_pos",
            "p_holm",
        },
        "check_low_n": True,
    },
    "reliability_ci_sensitivity.csv": {
        "required": {
            "method",
            "primary_cluster_by",
            "sensitivity_analysis",
            "signed_bias_ci_width_cluster",
            "signed_bias_ci_width_iid",
            "signed_bias_ci_width_ratio_cluster_over_iid",
            "pf_abs_err_ci_width_cluster",
            "pf_abs_err_ci_width_iid",
            "pf_abs_err_ci_width_ratio_cluster_over_iid",
        },
        "numeric": {
            "signed_bias_ci_width_cluster",
            "signed_bias_ci_width_iid",
            "signed_bias_ci_width_ratio_cluster_over_iid",
            "pf_abs_err_ci_width_cluster",
            "pf_abs_err_ci_width_iid",
            "pf_abs_err_ci_width_ratio_cluster_over_iid",
        },
        "expected_values": {
            "primary_cluster_by": {"case_id"},
            "sensitivity_analysis": {"iid_bootstrap"},
        },
    },
    "test_set_sufficiency.csv": {
        "required": {"case_id", "n_test", "nrmse_ci_width", "nrmse_relative_ci_width"},
        "numeric": {"n_test", "nrmse_ci_width", "nrmse_relative_ci_width"},
        "required_n_test": 1000,
    },
    "cross_metric_ranks.csv": {
        "required": {"method", "nrmse", "span"},
        "numeric": {"nrmse", "rmse", "mae", "maxae", "q95ae", "r2", "span"},
    },
    "cross_metric_rank_correlation.csv": {
        "required": {"metric", "nrmse"},
        "numeric": {"nrmse", "rmse", "mae", "maxae", "q95ae", "r2"},
    },
    "method_robustness.csv": {
        "required": {
            "method",
            "mean_nrmse",
            "mean_per_rep_std",
            "median_per_rep_std",
            "max_per_rep_std",
            "mean_cv",
        },
        "numeric": {
            "mean_nrmse",
            "mean_per_rep_std",
            "median_per_rep_std",
            "max_per_rep_std",
            "mean_cv",
        },
    },
}


def iter_candidate_files(root: Path) -> Iterable[Path]:
    """Yield every file under ``root`` whose extension is in
    :data:`EXTENSIONS` and whose path does not sit in a skip-dir.
    """
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix not in EXTENSIONS:
            continue
        if ".template" in p.name:
            continue
        if p.name in SKIP_BASENAMES:
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        yield p


def scan_file(path: Path) -> list[tuple[int, str, str]]:
    """Return ``(line_number, token_kind, line_content)`` unresolved hits."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Binary-ish file (rare under paper/); skip rather than crash.
        return []
    hits: list[tuple[int, str, str]] = []
    for i, line in enumerate(text.splitlines(), start=1):
        if _STUB_FALLBACK_RE.match(line):
            # Intentional \newcommand fallback stub — not a user-facing
            # placeholder; skip.
            continue
        for kind, pattern in _UNRESOLVED_PATTERNS:
            if pattern.search(line):
                hits.append((i, kind, line.strip()))
    return hits


def _markdown_body(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#") or stripped.startswith(">"):
            continue
        lines.append(line)
    return "\n".join(lines)


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text))


def _parse_iso_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None


def _normalize_distribution_name(name: str) -> str:
    return _DIST_NAME_NORMALIZE_RE.sub("-", str(name).strip().lower())


def _parse_pip_freeze_versions(path: Path) -> tuple[dict[str, str], list[str]]:
    """Parse `pip freeze --all` style `package==version` lines."""
    errors: list[str] = []
    versions: dict[str, str] = {}
    if path.is_symlink():
        return {}, [f"python environment lock is a symlink and cannot be trusted: {path}"]
    if not path.exists():
        return {}, [f"python environment lock is missing: {path}"]
    if not path.is_file() or path.stat().st_size == 0:
        return {}, [f"python environment lock is not a non-empty file: {path}"]

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PIP_FREEZE_VERSION_RE.match(line)
        if match is None:
            continue
        raw_name, version = match.groups()
        normalized = _normalize_distribution_name(raw_name)
        existing = versions.get(normalized)
        if existing is not None and existing != version:
            errors.append(
                "python environment lock contains conflicting versions for "
                f"{raw_name}: {existing!r} vs {version!r}"
            )
            continue
        versions[normalized] = version
    return versions, errors


def check_python_environment_lock_matches_run_config(
    run_config_path: Path,
    pip_freeze_path: Path,
) -> list[str]:
    """Ensure the archived pip-freeze lock matches the Stage-2 startup snapshot."""
    errors: list[str] = []
    if run_config_path.is_symlink():
        return [f"archived run_config.json is a symlink and cannot be trusted: {run_config_path}"]
    if not run_config_path.exists():
        return [f"archived run_config.json is missing: {run_config_path}"]
    if not run_config_path.is_file() or run_config_path.stat().st_size == 0:
        return [f"archived run_config.json is not a non-empty file: {run_config_path}"]
    try:
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [
            "archived run_config.json could not be parsed for python-environment "
            f"cross-check: {type(exc).__name__}: {exc}"
        ]

    environment = run_config.get("environment")
    if not isinstance(environment, dict):
        return ["archived run_config.json environment is missing or malformed"]
    packages = environment.get("packages")
    if not isinstance(packages, dict):
        return ["archived run_config.json environment.packages is missing or malformed"]

    pip_versions, pip_errors = _parse_pip_freeze_versions(pip_freeze_path)
    errors.extend(pip_errors)
    if pip_errors:
        return errors

    for package_name, payload in sorted(packages.items()):
        if not isinstance(payload, dict):
            errors.append(
                "archived run_config.json environment.packages "
                f"{package_name} is missing or malformed"
            )
            continue
        status = payload.get("status")
        version = payload.get("version")
        if status != "installed":
            errors.append(
                "archived run_config.json environment.packages "
                f"{package_name}.status must be 'installed' for the environment-lock "
                f"cross-check; got {status!r}"
            )
            continue
        if not isinstance(version, str) or not version.strip():
            errors.append(
                "archived run_config.json environment.packages "
                f"{package_name}.version is missing or malformed"
            )
            continue
        observed = pip_versions.get(_normalize_distribution_name(package_name))
        if observed is None:
            errors.append(
                "python environment lock does not record the run_config.json "
                f"startup package {package_name}"
            )
        elif observed != version:
            errors.append(
                "python environment lock version does not match run_config.json "
                f"for {package_name}: lock={observed!r}, run_config={version!r}"
            )
    return errors


def _read_json_object(path: Path, *, label: str) -> tuple[dict[str, object] | None, list[str]]:
    if path.is_symlink():
        return None, [f"{label} is a symlink and cannot be trusted: {path}"]
    if not path.exists():
        return None, [f"{label} is missing: {path}"]
    if not path.is_file() or path.stat().st_size == 0:
        return None, [f"{label} is not a non-empty file: {path}"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, [f"could not parse {label}: {type(exc).__name__}: {exc}"]
    if not isinstance(payload, dict):
        return None, [f"{label} root must be a JSON object: {path}"]
    return payload, []


def _normalize_storage_path(value: object) -> str:
    text = str(value or "").strip().replace("\\", "/")
    while "//" in text:
        text = text.replace("//", "/")
    return text.rstrip("/")


def check_manifest_temp_storage_matches_run_config(
    manifest_path: Path,
    run_config_path: Path,
) -> list[str]:
    """Bind RUN_MANIFEST scratch/tmp provenance to the Stage-2 startup snapshot."""
    manifest, manifest_errors = _read_json_object(manifest_path, label="RUN_MANIFEST.json")
    if manifest_errors:
        return manifest_errors
    run_config, run_config_errors = _read_json_object(run_config_path, label="archived run_config.json")
    if run_config_errors:
        return run_config_errors
    assert manifest is not None and run_config is not None

    machine = manifest.get("machine")
    if not isinstance(machine, dict):
        return ["RUN_MANIFEST machine section is missing or malformed"]
    manifest_temp = machine.get("temp_storage")
    if not isinstance(manifest_temp, dict):
        return ["RUN_MANIFEST machine.temp_storage is missing or malformed"]
    run_config_temp = run_config.get("temp_storage")
    if not isinstance(run_config_temp, dict):
        return ["archived run_config.json temp_storage is missing or malformed"]

    errors: list[str] = []
    manifest_path_value = _normalize_storage_path(manifest_temp.get("resolved_path"))
    run_config_path_value = _normalize_storage_path(run_config_temp.get("resolved_path"))
    if not manifest_path_value:
        errors.append("RUN_MANIFEST machine.temp_storage.resolved_path is missing or malformed")
    if not run_config_path_value:
        errors.append("archived run_config.json temp_storage.resolved_path is missing or malformed")
    if manifest_path_value and run_config_path_value and manifest_path_value != run_config_path_value:
        errors.append(
            "RUN_MANIFEST machine.temp_storage.resolved_path does not match "
            "archived run_config.json temp_storage.resolved_path: "
            f"manifest={manifest_path_value!r}, run_config={run_config_path_value!r}"
        )

    manifest_env = manifest_temp.get("env")
    run_config_env = run_config_temp.get("env")
    if not isinstance(manifest_env, dict):
        errors.append("RUN_MANIFEST machine.temp_storage.env is missing or malformed")
        manifest_env = {}
    if not isinstance(run_config_env, dict):
        errors.append("archived run_config.json temp_storage.env is missing or malformed")
        run_config_env = {}
    for name in TEMP_STORAGE_ENV_VARS:
        if name not in manifest_env:
            errors.append(f"RUN_MANIFEST machine.temp_storage.env missing {name}")
            continue
        if name not in run_config_env:
            errors.append(f"archived run_config.json temp_storage.env missing {name}")
            continue
        if manifest_env.get(name) != run_config_env.get(name):
            errors.append(
                "RUN_MANIFEST machine.temp_storage.env does not match archived "
                f"run_config.json temp_storage.env for {name}: "
                f"manifest={manifest_env.get(name)!r}, run_config={run_config_env.get(name)!r}"
            )
    return errors


def check_required_submission_sources(paper_dir: Path) -> list[str]:
    """Ensure templates have been materialised into real submission files."""
    errors: list[str] = []
    for rel in REQUIRED_SUBMISSION_SOURCES:
        path = paper_file(paper_dir, rel)
        if not path.is_file():
            errors.append(
                f"required submission source missing: {rel} "
                "(templates are not a submission-ready substitute)"
            )
    return errors


def check_editorial_submission_materials(paper_dir: Path) -> list[str]:
    """Validate Nature-family editorial-side submission files."""
    errors: list[str] = []

    significance = paper_file(paper_dir, "significance_statement.md")
    if significance.is_file():
        n_words = _word_count(_markdown_body(significance.read_text(encoding="utf-8")))
        if not (SIGNIFICANCE_WORD_MIN <= n_words <= SIGNIFICANCE_WORD_MAX):
            errors.append(
                "significance_statement.md body word count outside Nature-family "
                f"submission range {SIGNIFICANCE_WORD_MIN}-{SIGNIFICANCE_WORD_MAX}: "
                f"observed {n_words}"
            )

    cover_letter = paper_file(paper_dir, "cover_letter.md")
    if cover_letter.is_file():
        text = cover_letter.read_text(encoding="utf-8")
        n_words = _word_count(_markdown_body(text))
        if n_words > COVER_LETTER_WORD_MAX:
            errors.append(
                "cover_letter.md body word count exceeds Nature-family "
                f"submission limit {COVER_LETTER_WORD_MAX}: observed {n_words}"
            )
        for pattern, label in COVER_LETTER_REQUIRED_PATTERNS:
            if not re.search(pattern, text, flags=re.I | re.S):
                errors.append(f"cover_letter.md missing required element: {label}")
        number_macros = sorted(set(re.findall(r"\\Num[A-Za-z0-9]+", text)))
        if number_macros:
            errors.append(
                "cover_letter.md contains LaTeX number macros; write plain text "
                f"for editors: {number_macros}"
            )

    return errors


def _section_text(text: str, heading: str) -> str:
    pattern = rf"^##\s+{re.escape(heading)}\b.*?(?=^##\s+|\Z)"
    match = re.search(pattern, text, flags=re.I | re.M | re.S)
    return match.group(0) if match else ""


def _headline_decision(section: str) -> str | None:
    match = re.search(r"\*\*Headline decision\*\*\s*:\s*([^\n\r]+)", section, flags=re.I)
    if not match:
        return None
    decision = match.group(1).strip()
    decision = re.sub(r"[.;,]\s*$", "", decision)
    return decision


def _markdown_table_rows(section: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in section.splitlines():
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells):
            continue
        rows.append(cells)
    return rows


def _falsification_entry(section: str, discovery: str) -> str:
    pattern = rf"(?:^|\n)###\s+{re.escape(discovery)}\s+falsification\b.*?(?=\n###\s+|\Z)"
    match = re.search(pattern, section, flags=re.I | re.S)
    return match.group(0) if match else ""


def _manifest_artifact_sha(manifest_path: Path, artifact_name: str) -> tuple[str | None, str | None]:
    if not manifest_path.exists():
        return None, f"RUN_MANIFEST.json does not exist: {manifest_path}"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"could not parse RUN_MANIFEST.json: {type(exc).__name__}: {exc}"
    files = manifest.get("artifacts", {}).get("files")
    if not isinstance(files, list):
        return None, "RUN_MANIFEST.json artifacts.files is missing or malformed"
    matches = [
        item for item in files
        if isinstance(item, dict)
        and str(item.get("path", "")).replace("\\", "/").split("/")[-1] == artifact_name
    ]
    if not matches:
        return None, f"RUN_MANIFEST.json artifacts.files does not archive {artifact_name}"
    if len(matches) > 1:
        return None, f"RUN_MANIFEST.json artifacts.files has multiple {artifact_name} entries"
    sha = matches[0].get("sha256")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", sha.strip()):
        return None, f"RUN_MANIFEST.json {artifact_name} artifact lacks a valid sha256"
    return sha.strip().lower(), None


def _single_provenance_input_sha(
    payload: object,
    *,
    inputs_path: tuple[str, ...],
    label: str,
    context: str,
) -> tuple[str | None, list[str]]:
    """Return the unique provenance-input sha for ``label`` in a JSON payload."""
    errors: list[str] = []
    current = payload
    for key in inputs_path:
        if not isinstance(current, dict):
            return None, [f"{context}: missing {'.'.join(inputs_path)} block"]
        current = current.get(key)
    if not isinstance(current, list):
        return None, [f"{context}: {'.'.join(inputs_path)} is missing or malformed"]

    matches = [
        item for item in current
        if isinstance(item, dict) and str(item.get("label", "")).strip() == label
    ]
    if len(matches) != 1:
        return None, [
            f"{context}: expected exactly one provenance input labelled "
            f"{label!r}, observed {len(matches)}"
        ]
    sha = matches[0].get("sha256")
    if not isinstance(sha, str) or not _SHA256_RE.fullmatch(sha.strip()):
        return None, [
            f"{context}: provenance input {label!r} lacks a valid sha256"
        ]
    return sha.strip().lower(), errors


def check_paper_artifact_lineage(
    manifest_path: Path,
    numbers_path: Path,
    figdata_dir: Path,
) -> list[str]:
    """Ensure paper-derived artifacts are bound to the manifest's results.csv.

    ``check_numbers_provenance`` and ``check_figdata_manifest`` verify that each
    artifact matches its own recorded input hashes.  This cross-artifact gate
    adds the benchmark-submission invariant: paper numbers and figdata must have
    been generated from the same ``results.csv`` archived in ``RUN_MANIFEST``.
    """
    errors: list[str] = []
    manifest_results_sha, manifest_error = _manifest_artifact_sha(
        manifest_path,
        "results.csv",
    )
    if manifest_error:
        return [f"paper artifact lineage: {manifest_error}"]
    assert manifest_results_sha is not None

    try:
        numbers = json.loads(numbers_path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"paper artifact lineage: could not parse numbers.json: {type(exc).__name__}: {exc}")
    else:
        numbers_sha, sha_errors = _single_provenance_input_sha(
            numbers,
            inputs_path=("_provenance", "inputs"),
            label="results_csv",
            context="numbers.json",
        )
        errors.extend(sha_errors)
        if numbers_sha is not None and numbers_sha != manifest_results_sha:
            errors.append(
                "numbers.json results_csv provenance sha256 does not match "
                "RUN_MANIFEST archived results.csv: "
                f"numbers={numbers_sha}, manifest={manifest_results_sha}"
            )

    figdata_manifest = figdata_dir / "figdata_manifest.json"
    try:
        figdata = json.loads(figdata_manifest.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(
            "paper artifact lineage: could not parse figdata_manifest.json: "
            f"{type(exc).__name__}: {exc}"
        )
    else:
        figdata_sha, sha_errors = _single_provenance_input_sha(
            figdata,
            inputs_path=("inputs",),
            label="results_csv",
            context="figdata_manifest.json",
        )
        errors.extend(sha_errors)
        if figdata_sha is not None and figdata_sha != manifest_results_sha:
            errors.append(
                "figdata_manifest.json results_csv provenance sha256 does not "
                "match RUN_MANIFEST archived results.csv: "
                f"figdata={figdata_sha}, manifest={manifest_results_sha}"
            )

    return errors


def _manifest_decision_provenance(manifest_path: Path) -> tuple[dict[str, object], str | None]:
    if not manifest_path.exists():
        return {}, f"RUN_MANIFEST.json does not exist: {manifest_path}"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, f"could not parse RUN_MANIFEST.json: {type(exc).__name__}: {exc}"
    return {
        "started_at_utc": manifest.get("run_window", {}).get("started_at_utc")
        if isinstance(manifest.get("run_window"), dict)
        else None,
        "finished_at_utc": manifest.get("run_window", {}).get("finished_at_utc")
        if isinstance(manifest.get("run_window"), dict)
        else None,
        "machine_node": manifest.get("machine", {}).get("node")
        if isinstance(manifest.get("machine"), dict)
        else None,
    }, None


def _decision_results_sha(text: str) -> str | None:
    match = re.search(
        r"Manifest\s+sha256\s+for\s+results\.csv\*\*\s*:\s*`?([0-9a-fA-F]{64})`?",
        text,
        flags=re.I,
    )
    return match.group(1).lower() if match else None


def _decision_field(text: str, label: str) -> str | None:
    match = re.search(
        rf"^\s*-\s*\*\*{re.escape(label)}\*\*\s*:\s*(.+?)\s*$",
        text,
        flags=re.I | re.M,
    )
    if not match:
        return None
    return match.group(1).strip().strip("`")


def _decision_manifest_path(value: str | None) -> Path | None:
    if not _is_concrete_decision_value(value):
        return None
    assert value is not None
    raw = value.strip().strip("`")
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    return (ROOT / path).resolve()


def _is_concrete_decision_value(value: str | None) -> bool:
    if value is None:
        return False
    text = value.strip()
    if not text:
        return False
    if re.search(r"RESULT_PENDING|TBD|PLACEHOLDER|\[[^\]]+\]", text, flags=re.I):
        return False
    return True


def _is_iso_like_datetime(value: str | None) -> bool:
    if not _is_concrete_decision_value(value):
        return False
    assert value is not None
    return bool(
        re.search(
            r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:?\d{2})?\b",
            value,
        )
    )


def _is_utc8_signature_datetime(value: str | None) -> bool:
    parsed = _parse_iso_datetime(value)
    if parsed is None or parsed.tzinfo is None:
        return False
    return parsed.utcoffset() == timedelta(hours=8)


def _iso_datetimes_equal(a: str | None, b: object) -> bool:
    left = _parse_iso_datetime(a)
    right = _parse_iso_datetime(b)
    if left is None or right is None:
        return False
    return left == right


def check_preregistration_decisions(
    paper_dir: Path,
    manifest_path: Path | None = MANIFEST_DEFAULT,
) -> list[str]:
    """Validate the filled decision ledger before final submission."""
    path = paper_file(paper_dir, "preregistration_decisions.md")
    if not path.is_file():
        return ["preregistration_decisions.md missing; cannot validate signed decisions"]
    text = path.read_text(encoding="utf-8")
    errors: list[str] = []
    decisions: dict[str, str] = {}

    for label in ("D1", "D2", "D3"):
        section = _section_text(text, label)
        if not section:
            errors.append(f"preregistration_decisions.md missing {label} decision section")
            continue
        decision = _headline_decision(section)
        if decision is None:
            errors.append(f"preregistration_decisions.md {label} missing Headline decision")
        elif decision not in ALLOWED_DISCOVERY_DECISIONS:
            errors.append(
                f"preregistration_decisions.md {label} Headline decision must be "
                f"one of {sorted(ALLOWED_DISCOVERY_DECISIONS)}; observed {decision!r}"
            )
        else:
            decisions[label] = decision

    holm = _section_text(text, "Holm-Bonferroni")
    if not holm:
        errors.append("preregistration_decisions.md missing Holm-Bonferroni ledger")
    else:
        rows = _markdown_table_rows(holm)
        data_rows = [
            row for row in rows
            if row and row[0] in {"D1", "D2", "D3"}
        ]
        by_discovery = {row[0]: row for row in data_rows}
        missing = sorted({"D1", "D2", "D3"} - set(by_discovery))
        if missing:
            errors.append(
                "preregistration_decisions.md Holm-Bonferroni ledger missing "
                f"Discovery rows: {missing}"
            )
        for discovery, row in sorted(by_discovery.items()):
            pass_value = row[-1].strip() if row else ""
            if pass_value not in ALLOWED_BINARY_DECISIONS:
                errors.append(
                    "preregistration_decisions.md Holm-Bonferroni ledger "
                    f"{discovery} Pass? must be Y/N-like; observed {pass_value!r}"
                )

    falsification = _section_text(text, "Falsification log")
    needs_falsification_log = sorted(
        discovery
        for discovery, decision in decisions.items()
        if decision in {"Falsified", "Inconclusive"}
    )
    if needs_falsification_log and not falsification:
        errors.append(
            "preregistration_decisions.md missing Falsification log for "
            f"{needs_falsification_log}"
        )
    for discovery in needs_falsification_log:
        entry = _falsification_entry(falsification, discovery)
        if not entry:
            errors.append(
                "preregistration_decisions.md Falsification log missing "
                f"{discovery} falsification entry"
            )
            continue
        normalized_entry = re.sub(r"\s+", " ", entry)
        if not re.search(r"§\s*6|SAP\s+(?:Section\s+)?6|clause\s+6", normalized_entry, flags=re.I):
            errors.append(
                "preregistration_decisions.md Falsification log "
                f"{discovery} entry missing SAP Section 6 clause"
            )
        if not re.search(r"New\s+manuscript\s+headline\s*:\s*\S", normalized_entry, flags=re.I):
            errors.append(
                "preregistration_decisions.md Falsification log "
                f"{discovery} entry missing new manuscript headline"
            )

    if manifest_path is not None:
        decision_manifest = _decision_field(text, "Run manifest")
        decision_manifest_path = _decision_manifest_path(decision_manifest)
        expected_manifest_path = manifest_path.resolve()
        if decision_manifest_path is not None and decision_manifest_path != expected_manifest_path:
            errors.append(
                "preregistration_decisions.md Provenance Run manifest does not "
                f"match checked RUN_MANIFEST.json: observed {decision_manifest}, "
                f"expected {expected_manifest_path}"
            )

        manifest_provenance, provenance_error = _manifest_decision_provenance(manifest_path)
        if provenance_error:
            errors.append(
                "preregistration_decisions.md cannot validate manifest "
                f"provenance: {provenance_error}"
            )
        else:
            decision_started = _decision_field(text, "Sweep started (UTC)")
            expected_started = manifest_provenance.get("started_at_utc")
            if (
                _is_concrete_decision_value(decision_started)
                and expected_started
                and not _iso_datetimes_equal(decision_started, expected_started)
            ):
                errors.append(
                    "preregistration_decisions.md Provenance Sweep started (UTC) "
                    f"does not match RUN_MANIFEST run_window.started_at_utc: "
                    f"observed {decision_started}, expected {expected_started}"
                )
            decision_finished = _decision_field(text, "Sweep finished (UTC)")
            expected_finished = manifest_provenance.get("finished_at_utc")
            if (
                _is_concrete_decision_value(decision_finished)
                and expected_finished
                and not _iso_datetimes_equal(decision_finished, expected_finished)
            ):
                errors.append(
                    "preregistration_decisions.md Provenance Sweep finished (UTC) "
                    f"does not match RUN_MANIFEST run_window.finished_at_utc: "
                    f"observed {decision_finished}, expected {expected_finished}"
                )
            decision_host = _decision_field(text, "Cluster / host")
            expected_node = manifest_provenance.get("machine_node")
            if (
                _is_concrete_decision_value(decision_host)
                and isinstance(expected_node, str)
                and expected_node.strip()
                and expected_node.strip().lower() not in str(decision_host).lower()
            ):
                errors.append(
                    "preregistration_decisions.md Provenance Cluster / host "
                    f"does not mention RUN_MANIFEST machine.node: "
                    f"observed {decision_host}, expected node {expected_node}"
                )

        decision_sha = _decision_results_sha(text)
        if decision_sha is None:
            errors.append(
                "preregistration_decisions.md missing concrete "
                "Manifest sha256 for results.csv"
            )
        else:
            manifest_sha, manifest_error = _manifest_artifact_sha(
                manifest_path,
                "results.csv",
            )
            if manifest_error:
                errors.append(
                    "preregistration_decisions.md cannot validate results.csv "
                    f"manifest sha256: {manifest_error}"
                )
            elif decision_sha != manifest_sha:
                errors.append(
                    "preregistration_decisions.md Manifest sha256 for results.csv "
                    f"does not match RUN_MANIFEST.json: observed {decision_sha}, "
                    f"expected {manifest_sha}"
                )

    provenance_checks = (
        ("Run manifest", _is_concrete_decision_value, "concrete run-manifest path"),
        ("Sweep started (UTC)", _is_iso_like_datetime, "ISO-like sweep start time"),
        ("Sweep finished (UTC)", _is_iso_like_datetime, "ISO-like sweep finish time"),
        ("Cluster / host", _is_concrete_decision_value, "cluster/host record"),
        (
            "Author signing this decision document",
            _is_concrete_decision_value,
            "signing author identity",
        ),
        (
            "Date signed (UTC+08:00)",
            _is_utc8_signature_datetime,
            "UTC+08:00 signature datetime",
        ),
    )
    for label, validator, description in provenance_checks:
        value = _decision_field(text, label)
        if not validator(value):
            errors.append(
                "preregistration_decisions.md Provenance missing required "
                f"{description}: {label}"
            )

    signoff = _section_text(text, "Sign-off")
    if not signoff:
        errors.append("preregistration_decisions.md missing Sign-off section")
    else:
        normalized = re.sub(r"\s+", " ", signoff)
        required_signoff = (
            (r"numbers\s+above\s+are\s+read\s+directly\s+from\s+the\s+artefacts", "archive-artifact attestation"),
            (r"pinned\s+in\s+`?RUN_MANIFEST\.json`?", "RUN_MANIFEST pinning attestation"),
            (r"No\s+manuscript\s+text\s+was\s+amended\s+before\s+this\s+document\s+was\s+committed", "pre-manuscript-edit attestation"),
        )
        for pattern, label in required_signoff:
            if not re.search(pattern, normalized, flags=re.I):
                errors.append(
                    "preregistration_decisions.md Sign-off missing required "
                    f"{label}"
                )

    return errors


def check_manuscript_case_validation_caveat(paper_dir: Path) -> list[str]:
    """Ensure the manuscript does not overstate P2 case validation evidence."""
    path = paper_dir / "nature_cs_main.tex"
    if not path.is_file():
        return [
            "case-validation evidence-tier caveat cannot be checked: "
            "nature_cs_main.tex is missing"
        ]
    text = path.read_text(encoding="utf-8")
    normalized = re.sub(r"\s+", " ", text.replace("\\_", "_"))
    if all(pattern.search(normalized) for pattern in CASE_VALIDATION_CAVEAT_PATTERNS):
        return []
    return [
        "nature_cs_main.tex missing case-validation evidence-tier caveat: "
        "distinguish P2 plausibility-only cases from externally validated "
        "case-level claims"
    ]


def check_required_si_files(paper_dir: Path) -> list[str]:
    """Ensure auto-generated SI tables exist as submission artifacts."""
    errors: list[str] = []
    for rel in REQUIRED_SI_FILES:
        path = paper_file(paper_dir, rel)
        if not path.exists():
            errors.append(
                f"required SI artifact missing: {rel} "
                "(run scripts/generate_si.py before submission)"
            )
            continue
        if path.stat().st_size == 0:
            errors.append(f"required SI artifact is empty: {rel}")
    errors.extend(_check_si_case_evidence_tiers(paper_dir))
    return errors


def _check_si_case_evidence_tiers(paper_dir: Path) -> list[str]:
    """Ensure SI Table S3 exposes case-validation evidence tiers."""
    errors: list[str] = []
    md_path = paper_file(paper_dir, "si_cases.md")
    tex_path = paper_file(paper_dir, "si_cases.tex")
    if not md_path.exists() or not tex_path.exists():
        return errors
    md_text = md_path.read_text(encoding="utf-8")
    tex_text = tex_path.read_text(encoding="utf-8")

    if "Evidence tier" not in md_text:
        errors.append(
            "si_cases.md missing Evidence tier column; regenerate with "
            "python scripts/generate_si.py"
        )
    if r"\textbf{Evidence tier}" not in tex_text:
        errors.append(
            "si_cases.tex missing Evidence tier column; regenerate with "
            "python scripts/generate_si.py"
        )

    combined = (md_text + "\n" + tex_text).replace(r"\_", "_")
    missing_tiers = sorted(
        tier for tier in REQUIRED_CASE_EVIDENCE_TIERS if tier not in combined
    )
    if missing_tiers:
        errors.append(
            "si_cases evidence-tier inventory missing required tier(s) "
            f"{missing_tiers}; regenerate with python scripts/generate_si.py"
        )
    if "not_listed" in combined:
        errors.append(
            "si_cases evidence-tier inventory contains not_listed case(s); "
            "check docs/case_validation_status.csv and regenerate SI"
        )
    errors.extend(_check_si_case_evidence_tier_rows(md_text))
    return errors


def _expected_case_evidence_tiers() -> dict[str, str]:
    try:
        from scripts.generate_case_validation_summary import evidence_tier
    except Exception:
        return {}
    status_csv = CASE_VALIDATION_STATUS_DEFAULT
    if not status_csv.exists():
        return {}
    rows: dict[str, str] = {}
    with status_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            case_id = str(row.get("case_id", "")).strip()
            if case_id:
                rows[case_id] = evidence_tier(row)
    return rows


def _parse_si_case_evidence_tiers(md_text: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in md_text.splitlines():
        text = line.strip()
        if not text.startswith("|"):
            continue
        cells = [cell.strip() for cell in text.strip("|").split("|")]
        if len(cells) < 9 or cells[0].lower() in {"case", "---"}:
            continue
        case_id = cells[0].strip("` ")
        tier = cells[8].strip("` ")
        if case_id and tier:
            rows[case_id] = tier
    return rows


def _check_si_case_evidence_tier_rows(md_text: str) -> list[str]:
    expected = _expected_case_evidence_tiers()
    if not expected:
        return []
    observed = _parse_si_case_evidence_tiers(md_text)
    errors: list[str] = []
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    drift = [
        f"{case_id}: expected={expected[case_id]}, observed={observed[case_id]}"
        for case_id in sorted(set(expected) & set(observed))
        if observed[case_id] != expected[case_id]
    ]
    if missing:
        errors.append(
            "si_cases.md missing case evidence-tier row(s) "
            f"{missing[:10]}; regenerate with python scripts/generate_si.py"
        )
    if extra:
        errors.append(
            "si_cases.md contains unknown case evidence-tier row(s) "
            f"{extra[:10]}; check SI generation inputs"
        )
    if drift:
        errors.append(
            "si_cases.md evidence-tier drift vs docs/case_validation_status.csv: "
            f"{drift[:10]}; regenerate with python scripts/generate_si.py"
        )
    return errors


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _nonportable_command_file_requests(items: list[object]) -> list[str]:
    """Return command-history paths that cannot be restored repo-relatively."""
    bad: list[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text:
            continue
        normalized = text.replace("\\", "/")
        posix = PurePosixPath(normalized)
        windows = PureWindowsPath(text)
        if posix.is_absolute() or windows.is_absolute() or ".." in posix.parts:
            bad.append(item)
    return bad


def _split_command(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def _has_flag_value(tokens: list[str], flag: str, value: str) -> bool:
    return any(
        token == f"{flag}={value}"
        or (token == flag and idx + 1 < len(tokens) and tokens[idx + 1] == value)
        for idx, token in enumerate(tokens)
    )


def _formal_pipeline_invocation_error(command: str) -> str | None:
    tokens = _split_command(command)
    normalized = command.replace("\\", "/")
    uses_python_orchestrator = "scripts/run_pipeline.py" in normalized
    uses_shell_wrapper = "scripts/run_pipeline.sh" in normalized
    if not uses_python_orchestrator and not uses_shell_wrapper:
        return (
            "pipeline_invocation must use the formal full-pipeline entrypoint "
            "`scripts/run_pipeline.py` or `scripts/run_pipeline.sh`"
        )
    if "--dry-run" in tokens:
        return "pipeline_invocation must not be a dry-run command"
    if uses_python_orchestrator:
        if not _has_flag_value(tokens, "--profile", "full"):
            return "pipeline_invocation must use `--profile full`"
        if "--confirm-full-run" not in tokens:
            return "pipeline_invocation must include `--confirm-full-run`"
    elif _has_flag_value(tokens, "--profile", "smoke"):
        return "pipeline_invocation must not use the smoke profile"
    return None


def _repo_relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _current_case_source_hash(case_id: str) -> tuple[Path | None, str | None, str | None]:
    try:
        import doe_testbed as db

        case = db.get_case(case_id)
        module_name = type(case).__module__
        module = __import__(module_name, fromlist=["__file__"])
        source = Path(str(module.__file__)).resolve()
        return source, _sha256(source), None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def check_manifest_artifacts(manifest_path: Path) -> list[str]:
    """Validate artifact files listed in ``RUN_MANIFEST.json``.

    The manifest stores paths relative to the original results directory, which
    may have been generated on another host.  For a portable review archive, we
    interpret those paths relative to the directory containing the manifest.
    """
    errors: list[str] = []
    if not manifest_path.exists():
        return [f"RUN_MANIFEST.json does not exist: {manifest_path}"]

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"could not parse RUN_MANIFEST.json: {type(exc).__name__}: {exc}"]

    files = manifest.get("artifacts", {}).get("files")
    if not isinstance(files, list) or not files:
        return ["RUN_MANIFEST.json has no artifacts.files entries"]

    base_dir = manifest_path.parent.resolve()
    artifact_paths: list[str] = []
    for item in files:
        rel = item.get("path") if isinstance(item, dict) else None
        if not rel:
            errors.append(f"manifest artifact entry missing path: {item!r}")
            continue
        rel_text = str(rel).replace("\\", "/")
        artifact_paths.append(rel_text)
        rel_path = Path(str(rel))
        if rel_path.is_absolute() or ".." in rel_path.parts:
            errors.append(
                "manifest artifact path must be relative to RUN_MANIFEST.json "
                f"directory without parent traversal: {rel!r}"
            )
            continue
        raw_artifact = base_dir / rel_path
        if item.get("is_symlink") is True:
            errors.append(
                "manifest artifact entry records a symlink, which is not "
                f"acceptable for a submission archive: {rel!r}"
            )
            continue
        if raw_artifact.is_symlink():
            errors.append(
                "manifest artifact path is a symlink and cannot be trusted: "
                f"{raw_artifact}"
            )
            continue
        artifact = raw_artifact.resolve()
        try:
            artifact.relative_to(base_dir)
        except ValueError:
            errors.append(
                "manifest artifact path resolves outside RUN_MANIFEST.json "
                f"directory: {rel!r}"
            )
            continue
        if not artifact.exists():
            errors.append(f"manifest artifact missing: {artifact}")
            continue
        expected_size = item.get("size_bytes")
        if expected_size is not None:
            try:
                expected_size_int = int(expected_size)
            except (TypeError, ValueError):
                errors.append(
                    f"manifest artifact size_bytes is not an integer: {artifact} "
                    f"expected {expected_size!r}"
                )
            else:
                if artifact.stat().st_size != expected_size_int:
                    errors.append(
                        f"manifest artifact size mismatch: {artifact} "
                        f"expected {expected_size_int}, observed {artifact.stat().st_size}"
                    )
        expected_sha = item.get("sha256")
        if expected_sha:
            observed_sha = _sha256(artifact)
            if observed_sha.lower() != str(expected_sha).lower():
                errors.append(
                    f"manifest artifact sha256 mismatch: {artifact} "
                    f"expected {expected_sha}, observed {observed_sha}"
                )
    duplicates = sorted({rel for rel in artifact_paths if artifact_paths.count(rel) > 1})
    if duplicates:
        errors.append(f"RUN_MANIFEST artifacts.files contains duplicate paths: {duplicates}")
    return errors


def check_manifest_provenance(manifest_path: Path) -> list[str]:
    """Validate reproducibility-critical provenance fields in RUN_MANIFEST."""
    errors: list[str] = []
    if not manifest_path.exists():
        return [f"RUN_MANIFEST.json does not exist: {manifest_path}"]

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"could not parse RUN_MANIFEST.json: {type(exc).__name__}: {exc}"]

    generated_at = manifest.get("generated_at")
    if _parse_iso_datetime(generated_at) is None:
        errors.append("RUN_MANIFEST generated_at is missing or not ISO-8601")

    machine = manifest.get("machine")
    if not isinstance(machine, dict):
        errors.append("RUN_MANIFEST machine section is missing or malformed")
    else:
        if not machine.get("node"):
            errors.append("RUN_MANIFEST machine.node is missing")
        if not machine.get("platform"):
            errors.append("RUN_MANIFEST machine.platform is missing")
        if not machine.get("cpu_count"):
            errors.append("RUN_MANIFEST machine.cpu_count is missing")
        if machine.get("ram_gb") in (None, ""):
            errors.append("RUN_MANIFEST machine.ram_gb is missing")
        effective_ram = machine.get("effective_ram_gb")
        if effective_ram is not None:
            try:
                if float(effective_ram) <= 0:
                    errors.append("RUN_MANIFEST machine.effective_ram_gb must be positive when present")
            except (TypeError, ValueError):
                errors.append("RUN_MANIFEST machine.effective_ram_gb must be numeric when present")
        if not isinstance(machine.get("effective_ram_source"), str) or not machine.get("effective_ram_source", "").strip():
            errors.append("RUN_MANIFEST machine.effective_ram_source is missing or malformed")
        cgroup = machine.get("cgroup_memory")
        if not isinstance(cgroup, dict):
            errors.append("RUN_MANIFEST machine.cgroup_memory is missing or malformed")
        else:
            if cgroup.get("schema_version") != 1:
                errors.append("RUN_MANIFEST machine.cgroup_memory.schema_version must be 1")
            if not isinstance(cgroup.get("detected"), bool):
                errors.append("RUN_MANIFEST machine.cgroup_memory.detected must be boolean")
            if cgroup.get("limit_gb") is not None:
                try:
                    if float(cgroup.get("limit_gb")) <= 0:
                        errors.append("RUN_MANIFEST machine.cgroup_memory.limit_gb must be positive when present")
                except (TypeError, ValueError):
                    errors.append("RUN_MANIFEST machine.cgroup_memory.limit_gb must be numeric when present")
            if cgroup.get("unlimited") not in (True, False, None):
                errors.append("RUN_MANIFEST machine.cgroup_memory.unlimited must be boolean or null")
        resource_limits = machine.get("resource_limits")
        if not isinstance(resource_limits, dict):
            errors.append("RUN_MANIFEST machine.resource_limits is missing or malformed")
        else:
            if resource_limits.get("schema_version") != 1:
                errors.append("RUN_MANIFEST machine.resource_limits.schema_version must be 1")
            if not isinstance(resource_limits.get("status"), str) or not resource_limits.get("status", "").strip():
                errors.append("RUN_MANIFEST machine.resource_limits.status is missing or malformed")
            if not isinstance(resource_limits.get("limits"), dict):
                errors.append("RUN_MANIFEST machine.resource_limits.limits is missing or malformed")
        temp_storage = machine.get("temp_storage")
        if not isinstance(temp_storage, dict):
            errors.append("RUN_MANIFEST machine.temp_storage is missing or malformed")
        else:
            if temp_storage.get("schema_version") != 1:
                errors.append("RUN_MANIFEST machine.temp_storage.schema_version must be 1")
            resolved_path = temp_storage.get("resolved_path")
            if not isinstance(resolved_path, str) or not resolved_path.strip():
                errors.append("RUN_MANIFEST machine.temp_storage.resolved_path is missing or malformed")
            if not isinstance(temp_storage.get("exists"), bool):
                errors.append("RUN_MANIFEST machine.temp_storage.exists must be boolean")
            if not isinstance(temp_storage.get("writable"), bool):
                errors.append("RUN_MANIFEST machine.temp_storage.writable must be boolean")
            free_gb = temp_storage.get("free_gb")
            if free_gb is not None:
                try:
                    if float(free_gb) < 0:
                        errors.append("RUN_MANIFEST machine.temp_storage.free_gb must be non-negative when present")
                except (TypeError, ValueError):
                    errors.append("RUN_MANIFEST machine.temp_storage.free_gb must be numeric when present")
            if not isinstance(temp_storage.get("env"), dict):
                errors.append("RUN_MANIFEST machine.temp_storage.env is missing or malformed")

    python_environment = manifest.get("python_environment")
    if not isinstance(python_environment, dict):
        errors.append("RUN_MANIFEST python_environment section is missing or malformed")
    else:
        if python_environment.get("status") != "ok":
            errors.append(
                "RUN_MANIFEST python_environment.status must be 'ok'; "
                f"got {python_environment.get('status')!r}"
            )
        if not isinstance(python_environment.get("command"), str) or not python_environment.get("command", "").strip():
            errors.append("RUN_MANIFEST python_environment.command is missing or malformed")
        sha = python_environment.get("pip_freeze_sha256")
        if not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{64}", sha) is None:
            errors.append("RUN_MANIFEST python_environment.pip_freeze_sha256 is missing or invalid")
        package_count = python_environment.get("package_count")
        if not isinstance(package_count, int) or package_count <= 0:
            errors.append("RUN_MANIFEST python_environment.package_count must be a positive integer")
        pip_freeze_path = python_environment.get("pip_freeze_path")
        if not isinstance(pip_freeze_path, str) or not pip_freeze_path.strip():
            errors.append("RUN_MANIFEST python_environment.pip_freeze_path is missing or malformed")
        else:
            files = manifest.get("artifacts", {}).get("files")
            if not isinstance(files, list):
                errors.append("RUN_MANIFEST artifacts.files is missing or malformed")
            else:
                matches = [
                    item for item in files
                    if isinstance(item, dict)
                    and str(item.get("path", "")).replace("\\", "/") == pip_freeze_path.replace("\\", "/")
                ]
                if not matches:
                    errors.append(
                        "RUN_MANIFEST artifacts.files does not archive the python "
                        f"environment lock: {pip_freeze_path}"
                    )
                elif len(matches) > 1:
                    errors.append(
                        "RUN_MANIFEST artifacts.files archives the python environment "
                        f"lock more than once: {pip_freeze_path}"
                    )
                else:
                    match = matches[0]
                    artifact_sha = match.get("sha256")
                    if not isinstance(artifact_sha, str) or re.fullmatch(r"[0-9a-f]{64}", artifact_sha) is None:
                        errors.append(
                            "RUN_MANIFEST archived python environment lock lacks "
                            "a valid sha256"
                        )
                    elif artifact_sha.lower() != sha.lower():
                        errors.append(
                            "RUN_MANIFEST python_environment.pip_freeze_sha256 does "
                            "not match the archived python environment lock artifact"
                        )

    scheduler = manifest.get("scheduler")
    if not isinstance(scheduler, dict):
        errors.append("RUN_MANIFEST scheduler section is missing or malformed")
    else:
        if scheduler.get("schema_version") != 1:
            errors.append("RUN_MANIFEST scheduler.schema_version must be 1")
        if not isinstance(scheduler.get("name"), str) or not scheduler.get("name", "").strip():
            errors.append("RUN_MANIFEST scheduler.name is missing or malformed")
        if not isinstance(scheduler.get("detected"), bool):
            errors.append("RUN_MANIFEST scheduler.detected must be boolean")
        if not isinstance(scheduler.get("env"), dict):
            errors.append("RUN_MANIFEST scheduler.env is missing or malformed")

    accelerator = manifest.get("accelerator")
    if not isinstance(accelerator, dict):
        errors.append("RUN_MANIFEST accelerator section is missing or malformed")
    else:
        if accelerator.get("schema_version") != 1:
            errors.append("RUN_MANIFEST accelerator.schema_version must be 1")
        if not isinstance(accelerator.get("nvidia_smi"), dict):
            errors.append("RUN_MANIFEST accelerator.nvidia_smi is missing or malformed")
        torch_cuda = accelerator.get("torch_cuda")
        if not isinstance(torch_cuda, dict):
            errors.append("RUN_MANIFEST accelerator.torch_cuda is missing or malformed")
        elif torch_cuda.get("status") == "ok":
            if not isinstance(torch_cuda.get("cuda_available"), bool):
                errors.append("RUN_MANIFEST accelerator.torch_cuda.cuda_available must be boolean")
            if not isinstance(torch_cuda.get("device_count"), int) or torch_cuda.get("device_count") < 0:
                errors.append(
                    "RUN_MANIFEST accelerator.torch_cuda.device_count must be a non-negative integer"
                )

    container = manifest.get("container")
    if not isinstance(container, dict):
        errors.append("RUN_MANIFEST container section is missing or malformed")
    else:
        if container.get("schema_version") != 1:
            errors.append("RUN_MANIFEST container.schema_version must be 1")
        if not isinstance(container.get("detected"), bool):
            errors.append("RUN_MANIFEST container.detected must be boolean")
        if not isinstance(container.get("env"), dict):
            errors.append("RUN_MANIFEST container.env is missing or malformed")
        if not isinstance(container.get("marker_files"), dict):
            errors.append("RUN_MANIFEST container.marker_files is missing or malformed")

    run_window = manifest.get("run_window")
    if not isinstance(run_window, dict):
        errors.append("RUN_MANIFEST run_window section is missing or malformed")
    else:
        started_at = _parse_iso_datetime(run_window.get("started_at_utc"))
        finished_at = _parse_iso_datetime(run_window.get("finished_at_utc"))
        if started_at is None:
            errors.append("RUN_MANIFEST run_window.started_at_utc is missing or not ISO-8601")
        if finished_at is None:
            errors.append("RUN_MANIFEST run_window.finished_at_utc is missing or not ISO-8601")
        if started_at is not None and finished_at is not None and finished_at < started_at:
            errors.append("RUN_MANIFEST run_window.finished_at_utc is earlier than started_at_utc")
        n_artifacts = run_window.get("n_artifacts")
        if not isinstance(n_artifacts, int) or n_artifacts <= 0:
            errors.append("RUN_MANIFEST run_window.n_artifacts must be a positive integer")

    git = manifest.get("git", {})
    if not git.get("commit"):
        errors.append("RUN_MANIFEST git.commit is missing")
    if git.get("source_dirty"):
        status = git.get("source_dirty_entries") or []
        detail = f"; dirty source entries={len(status)}" if isinstance(status, list) else ""
        errors.append(f"RUN_MANIFEST was generated from a dirty source/config worktree{detail}")
    elif git.get("dirty") and "source_dirty" not in git:
        status = git.get("status_porcelain") or []
        detail = f"; dirty entries={len(status)}" if isinstance(status, list) else ""
        errors.append(f"RUN_MANIFEST was generated from a dirty git worktree{detail}")

    experiment = manifest.get("experiment", {})
    if not isinstance(experiment, dict):
        errors.append("RUN_MANIFEST experiment section is missing or malformed")
        experiment = {}

    commands = experiment.get("commands")
    case_resolution_error = experiment.get("case_resolution_error")
    if "case_resolution_error" not in experiment:
        errors.append("RUN_MANIFEST experiment.case_resolution_error is missing or malformed")
    elif case_resolution_error not in (None, ""):
        errors.append(
            f"RUN_MANIFEST experiment.case_resolution_error is non-empty: {case_resolution_error}"
        )
    if not isinstance(commands, list) or not commands:
        errors.append("RUN_MANIFEST experiment.commands is empty; capture the full reproduction command history")
    else:
        dry_run_commands = [
            command for command in commands
            if isinstance(command, str) and "--dry-run" in command.split()
        ]
        if dry_run_commands:
            errors.append(
                "RUN_MANIFEST experiment.commands contains --dry-run entries; "
                "capture commands from the real pipeline execution"
            )
    requested_command_files = experiment.get("command_files_requested")
    if not isinstance(requested_command_files, list):
        errors.append("RUN_MANIFEST experiment.command_files_requested is missing or malformed")
    elif not requested_command_files:
        errors.append("RUN_MANIFEST experiment.command_files_requested is empty; capture the full command history file")
    else:
        bad_requested = [
            item for item in requested_command_files
            if not isinstance(item, str) or not item.strip()
        ]
        if bad_requested:
            errors.append("RUN_MANIFEST experiment.command_files_requested contains blank or non-string entries")
        nonportable_requested = _nonportable_command_file_requests(requested_command_files)
        if nonportable_requested:
            errors.append(
                "RUN_MANIFEST experiment.command_files_requested contains non-portable paths; "
                f"use repo-relative paths without parent traversal: {nonportable_requested}"
            )
        dry_run_requested = [
            item for item in requested_command_files
            if isinstance(item, str) and ".dry_run" in item
        ]
        if dry_run_requested:
            errors.append(
                f"RUN_MANIFEST experiment.command_files_requested contains dry-run command history files: {dry_run_requested}"
            )
    missing_command_files = experiment.get("command_files_missing")
    if not isinstance(missing_command_files, list):
        errors.append("RUN_MANIFEST experiment.command_files_missing is missing or malformed")
    elif missing_command_files:
        errors.append(
            f"RUN_MANIFEST experiment.command_files_missing is non-empty: {missing_command_files}"
        )
    command_file_parse_errors = experiment.get("command_file_parse_errors")
    if not isinstance(command_file_parse_errors, list):
        errors.append("RUN_MANIFEST experiment.command_file_parse_errors is missing or malformed")
    elif command_file_parse_errors:
        errors.append(
            "RUN_MANIFEST experiment.command_file_parse_errors is non-empty: "
            f"{command_file_parse_errors}"
        )
    command_entries = experiment.get("command_entries")
    if not isinstance(command_entries, list) or not command_entries:
        errors.append("RUN_MANIFEST experiment.command_entries is missing or malformed")
    else:
        bad_entries = [
            item for item in command_entries
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("label"), str)
                or not item.get("label", "").strip()
                or not isinstance(item.get("command"), str)
                or not item.get("command", "").strip()
            )
        ]
        if bad_entries:
            errors.append("RUN_MANIFEST experiment.command_entries contains malformed entries")
        entry_commands = [
            item.get("command")
            for item in command_entries
            if isinstance(item, dict) and isinstance(item.get("command"), str)
        ]
        if isinstance(commands, list) and entry_commands != commands:
            errors.append("RUN_MANIFEST experiment.command_entries commands do not match experiment.commands")
        pipeline_invocations = [
            item for item in command_entries
            if (
                isinstance(item, dict)
                and item.get("label") == "pipeline_invocation"
                and isinstance(item.get("command"), str)
                and item.get("command", "").strip()
            )
        ]
        if not pipeline_invocations:
            errors.append("RUN_MANIFEST experiment.command_entries does not include pipeline_invocation")
        elif len(pipeline_invocations) > 1:
            errors.append("RUN_MANIFEST experiment.command_entries contains multiple pipeline_invocation entries")
        else:
            invocation_error = _formal_pipeline_invocation_error(
                str(pipeline_invocations[0].get("command", ""))
            )
            if invocation_error:
                errors.append(f"RUN_MANIFEST experiment.command_entries {invocation_error}")

    manifest_command = manifest.get("manifest_command")
    if not manifest_command:
        errors.append("RUN_MANIFEST manifest_command is missing")
    return errors


def _summary_nonnegative_int(
    summary: dict[str, object],
    key: str,
    errors: list[str],
) -> int:
    raw = summary.get(key, 0)
    if raw in (None, ""):
        return 0
    try:
        value = int(raw)
    except (TypeError, ValueError):
        errors.append(f"RUN_MANIFEST results_summary.{key} must be a non-negative integer")
        return 0
    if value < 0:
        errors.append(f"RUN_MANIFEST results_summary.{key} must be a non-negative integer")
        return 0
    return value


def _summary_status_count(value: object) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return None
    return parsed


def _manifest_int_list(value: object, label: str, errors: list[str]) -> list[int]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        errors.append(f"RUN_MANIFEST {label} must be a list of integers")
        return []
    out: list[int] = []
    for item in value:
        try:
            parsed = int(item)
        except (TypeError, ValueError):
            errors.append(f"RUN_MANIFEST {label} must be a list of integers")
            return []
        out.append(parsed)
    return sorted(out)


def _manifest_optional_nonnegative_int(
    value: object,
    label: str,
    errors: list[str],
) -> int | None:
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        errors.append(f"RUN_MANIFEST {label} must be a non-negative integer")
        return None
    if parsed < 0:
        errors.append(f"RUN_MANIFEST {label} must be a non-negative integer")
        return None
    return parsed


def _check_manifest_observed_string_set(
    *,
    summary: dict[str, object],
    declared: object,
    declared_count: object,
    observed_key: str,
    count_key: str,
    label: str,
    errors: list[str],
) -> None:
    observed = summary.get(observed_key)
    observed_count_raw = summary.get(count_key)
    observed_count = _manifest_optional_nonnegative_int(
        observed_count_raw,
        f"results_summary.{count_key}",
        errors,
    )
    if isinstance(declared_count, int) and observed_count_raw in (None, ""):
        errors.append(
            f"RUN_MANIFEST results_summary.{count_key} is missing or malformed"
        )
    if not isinstance(declared, list):
        if isinstance(declared_count, int) and observed_count != declared_count:
            errors.append(
                f"RUN_MANIFEST {label}-count drift: declared {declared_count}, "
                f"observed {summary.get(count_key)}"
            )
        return

    if not isinstance(observed, list):
        errors.append(
            f"RUN_MANIFEST results_summary.{observed_key} is missing or malformed"
        )
        return

    declared_values = [str(item) for item in declared]
    observed_values = [str(item) for item in observed]
    declared_set = set(declared_values)
    observed_set = set(observed_values)
    duplicates = sorted({
        item for item in observed_values if observed_values.count(item) > 1
    })
    missing = sorted(declared_set - observed_set)
    extra = sorted(observed_set - declared_set)
    if missing or extra:
        errors.append(
            f"RUN_MANIFEST {label} set drift: missing={missing}, extra={extra}"
        )
    if duplicates:
        errors.append(
            f"RUN_MANIFEST results_summary.{observed_key} contains duplicates: {duplicates}"
        )
    if observed_count is not None:
        if observed_count != len(observed_set):
            errors.append(
                f"RUN_MANIFEST results_summary.{count_key} does not match "
                f"the unique {label} observed set size"
            )
        if isinstance(declared_count, int) and observed_count != declared_count:
            errors.append(
                f"RUN_MANIFEST {label}-count drift: declared {declared_count}, "
                f"observed {observed_count}"
            )


def _current_results_summary_drift(df, summary: dict[str, object]) -> list[str]:
    """Compare RUN_MANIFEST results_summary against the current CSV contents."""
    errors: list[str] = []

    def _as_int_list(values: object) -> list[int] | None:
        if not isinstance(values, list):
            return None
        out: list[int] = []
        for item in values:
            try:
                out.append(int(float(str(item))))
            except (TypeError, ValueError):
                return None
        return sorted(out)

    actual_rows = int(len(df))
    if summary.get("n_rows") != actual_rows:
        errors.append(
            "RUN_MANIFEST results_summary.n_rows does not match current "
            f"results.csv row count: summary={summary.get('n_rows')!r}, "
            f"current={actual_rows}"
        )

    comparisons: list[tuple[str, str, object, object]] = []
    if "case_id" in df.columns:
        actual_cases = sorted(df["case_id"].dropna().astype(str).unique().tolist())
        comparisons.extend([
            ("case_ids_observed", "case set", summary.get("case_ids_observed"), actual_cases),
            ("n_cases_observed", "case count", summary.get("n_cases_observed"), len(actual_cases)),
        ])
    if "method" in df.columns:
        actual_methods = sorted(df["method"].dropna().astype(str).unique().tolist())
        comparisons.extend([
            ("methods_observed", "method set", summary.get("methods_observed"), actual_methods),
            ("n_methods_observed", "method count", summary.get("n_methods_observed"), len(actual_methods)),
        ])
    if "budget_mult" in df.columns:
        actual_budgets = sorted(
            int(value)
            for value in df["budget_mult"].dropna().astype(float).unique().tolist()
        )
        comparisons.append(
            ("budgets_observed", "budget set", _as_int_list(summary.get("budgets_observed")), actual_budgets)
        )
    if "repeat" in df.columns:
        actual_repeats = sorted(
            int(value)
            for value in df["repeat"].dropna().astype(float).unique().tolist()
        )
        comparisons.append(
            ("repeats_observed", "repeat set", _as_int_list(summary.get("repeats_observed")), actual_repeats)
        )
    if "status" in df.columns and isinstance(summary.get("status_counts"), dict):
        actual_status = {
            str(key): int(value)
            for key, value in df["status"].astype(str).value_counts().sort_index().items()
        }
        try:
            expected_status = {
                str(key): int(value)
                for key, value in summary.get("status_counts", {}).items()
            }
        except (TypeError, ValueError):
            expected_status = summary.get("status_counts")
        comparisons.append(("status_counts", "status counts", expected_status, actual_status))
    if (
        "failure_category" in df.columns
        and isinstance(summary.get("failure_category_counts"), dict)
    ):
        actual_failure_category = {
            str(key): int(value)
            for key, value in df["failure_category"]
            .fillna("")
            .astype(str)
            .value_counts()
            .sort_index()
            .items()
        }
        try:
            expected_failure_category = {
                str(key): int(value)
                for key, value in summary.get("failure_category_counts", {}).items()
            }
        except (TypeError, ValueError):
            expected_failure_category = summary.get("failure_category_counts")
        comparisons.append((
            "failure_category_counts",
            "failure-category counts",
            expected_failure_category,
            actual_failure_category,
        ))

    for key, label, expected, actual in comparisons:
        if expected != actual:
            errors.append(
                f"RUN_MANIFEST results_summary.{key} does not match current "
                f"results.csv {label}: summary={expected!r}, current={actual!r}"
            )
    return errors


def _finite_nonnegative_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed < 0:
        return None
    return parsed


def _check_runtime_resource_summary(
    summary: dict[str, object],
    current_results_df: object | None,
) -> list[str]:
    """Validate optional archived RSS resource roll-ups against results.csv."""
    resource = summary.get("runtime_resource_summary")
    if resource is None:
        return []
    if not isinstance(resource, dict):
        return ["RUN_MANIFEST results_summary.runtime_resource_summary is missing or malformed"]

    errors: list[str] = []
    if resource.get("rss_trace_columns_present") is not True:
        errors.append(
            "RUN_MANIFEST results_summary.runtime_resource_summary reports "
            f"missing RSS trace columns: {resource.get('missing_rss_trace_columns')}"
        )
    missing = resource.get("missing_rss_trace_columns")
    if not isinstance(missing, list):
        errors.append(
            "RUN_MANIFEST results_summary.runtime_resource_summary."
            "missing_rss_trace_columns is missing or malformed"
        )
    elif missing:
        errors.append(
            "RUN_MANIFEST results_summary.runtime_resource_summary has missing "
            f"RSS trace columns: {missing}"
        )

    resource_rows = _manifest_optional_nonnegative_int(
        resource.get("n_rows"),
        "results_summary.runtime_resource_summary.n_rows",
        errors,
    )
    valid_peak_rows = _manifest_optional_nonnegative_int(
        resource.get("n_rows_with_valid_rss_gb_peak"),
        "results_summary.runtime_resource_summary.n_rows_with_valid_rss_gb_peak",
        errors,
    )
    summary_rows = _manifest_optional_nonnegative_int(
        summary.get("n_rows"),
        "results_summary.n_rows",
        errors,
    )
    if summary_rows is not None and resource_rows != summary_rows:
        errors.append(
            "RUN_MANIFEST results_summary.runtime_resource_summary.n_rows "
            f"does not match results_summary.n_rows: resource={resource_rows}, "
            f"summary={summary_rows}"
        )
    if summary_rows is not None and valid_peak_rows != summary_rows:
        errors.append(
            "RUN_MANIFEST results_summary.runtime_resource_summary valid peak-RSS "
            f"row count drift: expected={summary_rows}, observed={valid_peak_rows}"
        )

    for field in (
        "rss_gb_peak_min",
        "rss_gb_peak_p50",
        "rss_gb_peak_p95",
        "rss_gb_peak_max",
    ):
        if valid_peak_rows:
            if _finite_nonnegative_float(resource.get(field)) is None:
                errors.append(
                    "RUN_MANIFEST results_summary.runtime_resource_summary."
                    f"{field} must be a finite non-negative number"
                )

    by_method = resource.get("rss_gb_peak_max_by_method")
    if by_method is not None:
        if not isinstance(by_method, dict):
            errors.append(
                "RUN_MANIFEST results_summary.runtime_resource_summary."
                "rss_gb_peak_max_by_method is missing or malformed"
            )
        else:
            bad_methods = [
                str(method)
                for method, value in by_method.items()
                if _finite_nonnegative_float(value) is None
            ]
            if bad_methods:
                errors.append(
                    "RUN_MANIFEST results_summary.runtime_resource_summary has "
                    f"non-finite method peak RSS values for {sorted(bad_methods)}"
                )

    if current_results_df is not None and hasattr(current_results_df, "columns"):
        df = current_results_df
        if "rss_gb_peak" in df.columns:
            import pandas as pd

            peaks = pd.to_numeric(df["rss_gb_peak"], errors="coerce")
            valid_peaks = peaks.dropna()
            if valid_peak_rows != int(len(valid_peaks)):
                errors.append(
                    "RUN_MANIFEST results_summary.runtime_resource_summary "
                    "n_rows_with_valid_rss_gb_peak does not match current "
                    f"results.csv: summary={valid_peak_rows}, current={len(valid_peaks)}"
                )
            if len(valid_peaks):
                current_max = round(float(valid_peaks.max()), 6)
                archived_max = _finite_nonnegative_float(resource.get("rss_gb_peak_max"))
                if archived_max is None or round(archived_max, 6) != current_max:
                    errors.append(
                        "RUN_MANIFEST results_summary.runtime_resource_summary "
                        "rss_gb_peak_max does not match current results.csv: "
                        f"summary={resource.get('rss_gb_peak_max')!r}, "
                        f"current={current_max}"
                    )
    return errors


def _check_runtime_timestamp_summary(
    summary: dict[str, object],
    current_results_df: object | None,
) -> list[str]:
    """Validate archived per-cell runtime timestamp roll-ups."""
    timestamp = summary.get("runtime_timestamp_summary")
    if timestamp is None:
        return []
    if not isinstance(timestamp, dict):
        return ["RUN_MANIFEST results_summary.runtime_timestamp_summary is missing or malformed"]

    errors: list[str] = []
    if timestamp.get("timestamp_trace_columns_present") is not True:
        errors.append(
            "RUN_MANIFEST results_summary.runtime_timestamp_summary reports "
            "missing timestamp trace columns: "
            f"{timestamp.get('missing_timestamp_trace_columns')}"
        )
    missing = timestamp.get("missing_timestamp_trace_columns")
    if not isinstance(missing, list):
        errors.append(
            "RUN_MANIFEST results_summary.runtime_timestamp_summary."
            "missing_timestamp_trace_columns is missing or malformed"
        )
    elif missing:
        errors.append(
            "RUN_MANIFEST results_summary.runtime_timestamp_summary has missing "
            f"timestamp trace columns: {missing}"
        )

    timestamp_rows = _manifest_optional_nonnegative_int(
        timestamp.get("n_rows"),
        "results_summary.runtime_timestamp_summary.n_rows",
        errors,
    )
    summary_rows = _manifest_optional_nonnegative_int(
        summary.get("n_rows"),
        "results_summary.n_rows",
        errors,
    )
    if summary_rows is not None and timestamp_rows != summary_rows:
        errors.append(
            "RUN_MANIFEST results_summary.runtime_timestamp_summary.n_rows "
            f"does not match results_summary.n_rows: timestamp={timestamp_rows}, "
            f"summary={summary_rows}"
        )

    for key in (
        "n_invalid_runtime_timestamp_values",
        "n_invalid_runtime_time_order_values",
        "n_invalid_runtime_duration_values",
    ):
        value = _manifest_optional_nonnegative_int(
            timestamp.get(key),
            f"results_summary.runtime_timestamp_summary.{key}",
            errors,
        )
        summary_value = _manifest_optional_nonnegative_int(
            summary.get(key),
            f"results_summary.{key}",
            errors,
        )
        if value is not None and summary_value is not None and value != summary_value:
            errors.append(
                "RUN_MANIFEST results_summary.runtime_timestamp_summary "
                f"{key} does not match top-level summary: "
                f"timestamp={value}, summary={summary_value}"
            )

    if current_results_df is not None and hasattr(current_results_df, "columns"):
        df = current_results_df
        if {"cell_started_utc", "cell_finished_utc"}.issubset(df.columns):
            import pandas as pd

            started = pd.to_datetime(
                df["cell_started_utc"].fillna("").astype(str).str.strip(),
                errors="coerce",
                utc=True,
            )
            finished = pd.to_datetime(
                df["cell_finished_utc"].fillna("").astype(str).str.strip(),
                errors="coerce",
                utc=True,
            )
            invalid_timestamp_values = int(started.isna().sum() + finished.isna().sum())
            invalid_time_order = int((finished < started).sum())
            invalid_duration = 0
            if "t_total_s" in df.columns:
                invalid_duration = int(
                    invalid_runtime_duration_mask(
                        df["t_total_s"],
                        started,
                        finished,
                    ).sum()
                )
            archived_invalid_timestamps = _manifest_optional_nonnegative_int(
                timestamp.get("n_invalid_runtime_timestamp_values"),
                "results_summary.runtime_timestamp_summary.n_invalid_runtime_timestamp_values",
                errors,
            )
            archived_invalid_order = _manifest_optional_nonnegative_int(
                timestamp.get("n_invalid_runtime_time_order_values"),
                "results_summary.runtime_timestamp_summary.n_invalid_runtime_time_order_values",
                errors,
            )
            archived_invalid_duration = _manifest_optional_nonnegative_int(
                timestamp.get("n_invalid_runtime_duration_values"),
                "results_summary.runtime_timestamp_summary.n_invalid_runtime_duration_values",
                errors,
            )
            if archived_invalid_timestamps != invalid_timestamp_values:
                errors.append(
                    "RUN_MANIFEST results_summary.runtime_timestamp_summary "
                    "n_invalid_runtime_timestamp_values does not match current "
                    f"results.csv: summary={archived_invalid_timestamps}, "
                    f"current={invalid_timestamp_values}"
                )
            if archived_invalid_order != invalid_time_order:
                errors.append(
                    "RUN_MANIFEST results_summary.runtime_timestamp_summary "
                    "n_invalid_runtime_time_order_values does not match current "
                    f"results.csv: summary={archived_invalid_order}, "
                    f"current={invalid_time_order}"
                )
            if archived_invalid_duration != invalid_duration:
                errors.append(
                    "RUN_MANIFEST results_summary.runtime_timestamp_summary "
                    "n_invalid_runtime_duration_values does not match current "
                    f"results.csv: summary={archived_invalid_duration}, "
                    f"current={invalid_duration}"
                )
    return errors


def check_manifest_result_contract(manifest_path: Path) -> list[str]:
    """Validate the internal consistency of RUN_MANIFEST result metadata."""
    errors: list[str] = []
    if not manifest_path.exists():
        return [f"RUN_MANIFEST.json does not exist: {manifest_path}"]

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"could not parse RUN_MANIFEST.json: {type(exc).__name__}: {exc}"]

    experiment = manifest.get("experiment", {})
    summary = manifest.get("results_summary")
    if not isinstance(summary, dict):
        return ["RUN_MANIFEST results_summary is missing or malformed"]
    if summary.get("exists") is not True:
        errors.append("RUN_MANIFEST results_summary says results.csv was not present when archived")

    artifacts = manifest.get("artifacts", {})
    files = artifacts.get("files") if isinstance(artifacts, dict) else None
    if not isinstance(files, list):
        errors.append("RUN_MANIFEST artifacts.files is missing or malformed")
        artifact_paths: set[str] = set()
    else:
        artifact_paths = {
            str(item.get("path")).replace("\\", "/")
            for item in files
            if isinstance(item, dict) and item.get("path")
        }
    if "results.csv" not in artifact_paths:
        errors.append("RUN_MANIFEST artifacts.files does not archive results.csv")
    if "sweep_integrity_report.md" not in artifact_paths:
        errors.append("RUN_MANIFEST artifacts.files does not archive sweep_integrity_report.md")
    python_environment = manifest.get("python_environment")
    pip_freeze_path = (
        python_environment.get("pip_freeze_path")
        if isinstance(python_environment, dict)
        else None
    )
    if isinstance(pip_freeze_path, str) and pip_freeze_path.strip():
        normalized_pip_freeze_path = pip_freeze_path.replace("\\", "/")
        if normalized_pip_freeze_path not in artifact_paths:
            errors.append(
                "RUN_MANIFEST artifacts.files does not archive the python "
                f"environment lock: {normalized_pip_freeze_path}"
            )
        else:
            pip_freeze_file = manifest_path.parent / Path(pip_freeze_path)
            if not pip_freeze_file.exists():
                errors.append(
                    "RUN_MANIFEST archived python environment lock is missing beside "
                    f"manifest: {pip_freeze_file}"
                )
            else:
                run_config_file = manifest_path.parent / "run_config.json"
                if run_config_file.exists():
                    errors.extend(
                        check_python_environment_lock_matches_run_config(
                            run_config_file,
                            pip_freeze_file,
                        )
                    )

    run_config_file = manifest_path.parent / "run_config.json"
    if run_config_file.exists():
        errors.extend(
            check_manifest_temp_storage_matches_run_config(
                manifest_path,
                run_config_file,
            )
        )

    results_csv = manifest_path.parent / "results.csv"
    current_results_df = None
    if summary.get("exists") is True and not results_csv.exists():
        errors.append(f"RUN_MANIFEST archived results.csv is missing beside manifest: {results_csv}")
    elif results_csv.exists():
        try:
            import pandas as pd
            from scripts.prepare_figdata import (
                validate_headline_results_source,
                validate_results_contract,
            )

            df_current = pd.read_csv(results_csv, low_memory=False)
            validate_results_contract(df_current)
            current_results_df = df_current
            if manifest_path.resolve() == MANIFEST_DEFAULT.resolve():
                validate_headline_results_source(results_csv, df_current)
        except AssertionError as exc:
            errors.append(f"RUN_MANIFEST current results.csv row-level contract failed: {exc}")
        except Exception as exc:
            errors.append(
                "RUN_MANIFEST current results.csv could not be validated: "
                f"{type(exc).__name__}: {exc}"
            )

    observed_rows = _manifest_optional_nonnegative_int(
        summary.get("n_rows"),
        "results_summary.n_rows",
        errors,
    )
    budgets = _manifest_int_list(
        experiment.get("budgets", []),
        "experiment.budgets",
        errors,
    )
    n_cases = experiment.get("n_cases_declared")
    n_methods = experiment.get("n_methods_declared")
    n_repeats = _manifest_optional_nonnegative_int(
        experiment.get("n_repeats"),
        "experiment.n_repeats",
        errors,
    )
    expected_total = None
    if all(isinstance(v, int) for v in [n_cases, n_methods, n_repeats]) and budgets:
        expected_total = int(n_cases) * int(n_methods) * len(budgets) * int(n_repeats)
        if observed_rows is not None and observed_rows != expected_total:
            errors.append(
                f"RUN_MANIFEST result grid mismatch: declared grid has {expected_total} cells "
                f"but results_summary.n_rows is {observed_rows}"
            )

    declared_cases = experiment.get("case_ids_declared")
    _check_manifest_observed_string_set(
        summary=summary,
        declared=declared_cases,
        declared_count=n_cases,
        observed_key="case_ids_observed",
        count_key="n_cases_observed",
        label="case",
        errors=errors,
    )

    declared_methods = experiment.get("methods_declared")
    _check_manifest_observed_string_set(
        summary=summary,
        declared=declared_methods,
        declared_count=n_methods,
        observed_key="methods_observed",
        count_key="n_methods_observed",
        label="method",
        errors=errors,
    )

    observed_budgets_raw = summary.get("budgets_observed")
    if not isinstance(observed_budgets_raw, list):
        errors.append(
            "RUN_MANIFEST results_summary.budgets_observed is missing or malformed"
        )
    observed_budgets = _manifest_int_list(
        observed_budgets_raw,
        "results_summary.budgets_observed",
        errors,
    )
    if budgets and observed_budgets:
        expected_budgets = sorted(budgets)
        if observed_budgets != expected_budgets:
            errors.append(
                f"RUN_MANIFEST budget set drift: expected {expected_budgets}, "
                f"observed {observed_budgets_raw}"
            )

    observed_repeats_raw = summary.get("repeats_observed")
    if not isinstance(observed_repeats_raw, list):
        errors.append(
            "RUN_MANIFEST results_summary.repeats_observed is missing or malformed"
        )
    observed_repeats = _manifest_int_list(
        observed_repeats_raw,
        "results_summary.repeats_observed",
        errors,
    )
    if isinstance(n_repeats, int) and observed_repeats:
        expected_repeats = list(range(n_repeats))
        if observed_repeats != expected_repeats:
            errors.append(
                f"RUN_MANIFEST repeat set drift: expected {expected_repeats}, "
                f"observed {observed_repeats_raw}"
            )

    status_counts = summary.get("status_counts", {})
    if not isinstance(status_counts, dict):
        errors.append("RUN_MANIFEST status_counts is missing or malformed")
    else:
        status_total = 0
        status_total_valid = True
        unknown_statuses: list[str] = []
        for key, value in status_counts.items():
            if str(key) not in {"ok", "timeout", "failed"}:
                unknown_statuses.append(str(key))
            parsed = _summary_status_count(value)
            if parsed is None:
                errors.append(
                    f"RUN_MANIFEST status_counts has non-negative integer count required "
                    f"for {key!r}: {value!r}"
                )
                status_total_valid = False
            else:
                status_total += parsed
        if unknown_statuses:
            errors.append(f"RUN_MANIFEST status_counts contains unknown statuses: {sorted(unknown_statuses)}")
        if status_total_valid and summary.get("n_rows") is not None and status_total != summary.get("n_rows"):
            errors.append(
                f"RUN_MANIFEST status_counts total {status_total} does not match "
                f"results_summary.n_rows {summary.get('n_rows')}"
            )

    failure_category_counts = summary.get("failure_category_counts", {})
    if not isinstance(failure_category_counts, dict):
        errors.append("RUN_MANIFEST failure_category_counts is missing or malformed")
    else:
        category_total = 0
        category_total_valid = True
        unknown_categories: list[str] = []
        for key, value in failure_category_counts.items():
            if str(key) not in ALLOWED_FAILURE_CATEGORIES:
                unknown_categories.append(str(key))
            parsed = _summary_status_count(value)
            if parsed is None:
                errors.append(
                    "RUN_MANIFEST failure_category_counts has non-negative "
                    f"integer count required for {key!r}: {value!r}"
                )
                category_total_valid = False
            else:
                category_total += parsed
        if unknown_categories:
            errors.append(
                "RUN_MANIFEST failure_category_counts contains unknown "
                f"categories: {sorted(unknown_categories)}"
            )
        if (
            category_total_valid
            and summary.get("n_rows") is not None
            and category_total != summary.get("n_rows")
        ):
            errors.append(
                f"RUN_MANIFEST failure_category_counts total {category_total} "
                f"does not match results_summary.n_rows {summary.get('n_rows')}"
            )

    if current_results_df is not None:
        errors.extend(_current_results_summary_drift(current_results_df, summary))

    n_bad_budgets = _summary_nonnegative_int(
        summary, "n_invalid_budget_mult_values", errors
    )
    if n_bad_budgets:
        errors.append(
            f"RUN_MANIFEST results_summary has {n_bad_budgets} invalid budget_mult values"
        )
    n_bad_repeats = _summary_nonnegative_int(
        summary, "n_invalid_repeat_values", errors
    )
    if n_bad_repeats:
        errors.append(
            f"RUN_MANIFEST results_summary has {n_bad_repeats} invalid repeat values"
        )
    row_contract_fields = {
        "n_missing_protocol_key_values": "rows with missing protocol key values",
        "n_invalid_status_values": "invalid status values",
        "n_missing_failure_category_values": "rows with missing failure categories",
        "n_invalid_failure_category_values": "invalid failure category values",
        "n_failure_category_status_mismatch_values": "rows with failure category/status mismatches",
        "n_invalid_ok_nrmse_values": "invalid ok-row nrmse values",
        "n_invalid_t_total_s_values": "invalid t_total_s values",
        "n_invalid_runtime_timestamp_values": "invalid runtime timestamp values",
        "n_invalid_runtime_time_order_values": "rows with runtime finish before start",
        "n_invalid_runtime_duration_values": "rows with t_total_s inconsistent with runtime timestamps",
        "n_invalid_rss_gb_values": "invalid RSS trace values",
        "n_invalid_rss_gb_peak_order_values": "rows with peak RSS below start/end RSS",
        "n_missing_failure_error_values": "timeout/failed rows without error reasons",
        "n_ok_rows_with_error_values": "ok rows with non-empty errors",
    }
    for key, label in row_contract_fields.items():
        value = _summary_nonnegative_int(summary, key, errors)
        if value:
            errors.append(f"RUN_MANIFEST results_summary has {value} {label}")
    errors.extend(_check_runtime_timestamp_summary(summary, current_results_df))
    errors.extend(_check_runtime_resource_summary(summary, current_results_df))
    return errors


def check_case_source_fingerprints(manifest_path: Path) -> list[str]:
    """Validate that RUN_MANIFEST case source hashes match the current tree."""
    errors: list[str] = []
    if not manifest_path.exists():
        return [f"RUN_MANIFEST.json does not exist: {manifest_path}"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"could not parse RUN_MANIFEST.json: {type(exc).__name__}: {exc}"]

    declared = manifest.get("experiment", {}).get("case_ids_declared")
    if not isinstance(declared, list) or not declared:
        return errors
    case_sources = manifest.get("case_sources")
    if not isinstance(case_sources, dict):
        return ["RUN_MANIFEST case_sources fingerprints are missing"]

    entries = case_sources.get("cases")
    if not isinstance(entries, list) or not entries:
        return ["RUN_MANIFEST case_sources.cases is empty"]

    by_case = {
        str(item.get("case_id")): item
        for item in entries
        if isinstance(item, dict) and item.get("case_id")
    }
    for case_id in declared:
        item = by_case.get(str(case_id))
        if item is None:
            errors.append(f"RUN_MANIFEST missing case source fingerprint for {case_id}")
            continue
        if item.get("status") != "ok":
            errors.append(
                f"RUN_MANIFEST case source fingerprint unresolved for {case_id}: "
                f"{item.get('error', item.get('status'))}"
            )
            continue
        expected_sha = item.get("sha256")
        expected_source = item.get("source_path")
        source, observed_sha, error = _current_case_source_hash(str(case_id))
        if error:
            errors.append(f"current case source unresolved for {case_id}: {error}")
            continue
        observed_source = _repo_relative_path(source) if source else "<unknown>"
        if str(expected_source).replace("\\", "/") != observed_source:
            errors.append(
                f"case source path mismatch for {case_id}: "
                f"expected {observed_source}, observed {expected_source!r}"
            )
        if observed_sha != expected_sha:
            errors.append(
                f"case source hash mismatch for {case_id}: {observed_source} "
                f"expected {expected_sha}, observed {observed_sha}"
            )
    return errors


def check_method_source_fingerprints(manifest_path: Path) -> list[str]:
    """Validate RUN_MANIFEST method source fingerprints against current code."""
    errors: list[str] = []
    if not manifest_path.exists():
        return [f"RUN_MANIFEST.json does not exist: {manifest_path}"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"could not parse RUN_MANIFEST.json: {type(exc).__name__}: {exc}"]

    experiment = manifest.get("experiment", {})
    pool = experiment.get("pool")
    declared = experiment.get("methods_declared")
    if not pool or not isinstance(declared, list) or not declared:
        return errors
    method_sources = manifest.get("method_sources")
    if not isinstance(method_sources, dict):
        return ["RUN_MANIFEST method_sources fingerprints are missing"]
    entries = method_sources.get("methods")
    if not isinstance(entries, list) or not entries:
        return ["RUN_MANIFEST method_sources.methods is empty"]

    by_method = {
        str(item.get("method")): item
        for item in entries
        if isinstance(item, dict) and item.get("method")
    }
    try:
        from benchmark.core import method_source_components, method_source_fingerprint
        from benchmark.scripts.run_full_benchmark import POOLS

        current_methods = {m.name: m for m in POOLS[str(pool)]()}
    except Exception as exc:
        return [f"current method source fingerprints unresolved for pool={pool!r}: {type(exc).__name__}: {exc}"]

    for method_name in declared:
        method_name = str(method_name)
        item = by_method.get(method_name)
        if item is None:
            errors.append(f"RUN_MANIFEST missing method source fingerprint for {method_name}")
            continue
        if item.get("status") != "ok":
            errors.append(
                f"RUN_MANIFEST method source fingerprint unresolved for {method_name}: "
                f"{item.get('error', item.get('status'))}"
            )
            continue
        method = current_methods.get(method_name)
        if method is None:
            errors.append(f"current method pool={pool!r} no longer exposes {method_name}")
            continue
        current = method_source_fingerprint(method)
        expected_path = current["method_source_path"]
        observed_path = str(item.get("method_source_path", "")).replace("\\", "/")
        if observed_path != expected_path:
            errors.append(
                f"method source path mismatch for {method_name}: "
                f"expected {expected_path}, observed {observed_path!r}"
            )
        expected_sha = current["method_source_sha256"]
        observed_sha = item.get("method_source_sha256")
        if observed_sha != expected_sha:
            errors.append(
                f"method source hash mismatch for {method_name}: "
                f"expected {expected_sha}, observed {observed_sha}"
            )
        expected_components = method_source_components(method)
        observed_components = item.get("components")
        if not isinstance(observed_components, list) or not observed_components:
            errors.append(
                f"RUN_MANIFEST method source components are missing for {method_name}"
            )
        elif observed_components != expected_components:
            errors.append(
                f"method source components mismatch for {method_name}: "
                "manifest does not match current source/config components"
            )
    return errors


def check_case_validation_status(status_csv: Path) -> list[str]:
    """Validate the case correctness audit ledger used for submission."""
    errors: list[str] = []
    if not status_csv.exists():
        return [f"case validation status CSV does not exist: {status_csv}"]

    try:
        with status_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            columns = list(reader.fieldnames or [])
            rows = list(reader)
    except Exception as exc:
        return [f"could not parse case validation status CSV: {type(exc).__name__}: {exc}"]

    if columns != CASE_VALIDATION_REQUIRED_COLUMNS:
        errors.append(
            "case validation status schema mismatch: "
            f"expected {CASE_VALIDATION_REQUIRED_COLUMNS}, observed {columns}"
        )

    try:
        import doe_testbed as db

        metas = {meta.case_id: meta for meta in db.list_cases()}
    except Exception as exc:
        return errors + [f"could not load registered case metadata: {type(exc).__name__}: {exc}"]

    observed = [row.get("case_id", "") for row in rows]
    observed_set = {case_id for case_id in observed if case_id}
    registered = set(metas)
    missing = sorted(registered - observed_set)
    extra = sorted(observed_set - registered)
    duplicates = sorted({case_id for case_id in observed if observed.count(case_id) > 1})
    if missing:
        errors.append(f"case validation status missing registered cases: {missing}")
    if extra:
        errors.append(f"case validation status contains unknown cases: {extra}")
    if duplicates:
        errors.append(f"case validation status duplicates cases: {duplicates}")
    if len(rows) != PREREG_N_CASES:
        errors.append(
            f"case validation status row count mismatch: expected {PREREG_N_CASES}, observed {len(rows)}"
        )

    validation_root = ROOT / "案例正确性校验"
    documented_note_dirs: set[Path] = set()
    for row in rows:
        case_id = row.get("case_id", "")
        if case_id in metas:
            meta = metas[case_id]
            if row.get("domain") != meta.domain:
                errors.append(
                    f"{case_id}: domain mismatch in case validation status: "
                    f"expected {meta.domain}, observed {row.get('domain')!r}"
                )
            try:
                dim = int(str(row.get("dim", "")))
            except ValueError:
                errors.append(f"{case_id}: dim is not an integer in case validation status")
            else:
                if dim != int(meta.dim_input):
                    errors.append(
                        f"{case_id}: dim mismatch in case validation status: "
                        f"expected {meta.dim_input}, observed {dim}"
                    )

        risk = row.get("risk_level", "")
        if risk not in CASE_VALIDATION_ALLOWED_RISKS:
            errors.append(
                f"{case_id}: risk_level {risk!r} is not submission-ready; "
                f"allowed values are {sorted(CASE_VALIDATION_ALLOWED_RISKS)}"
            )
        final_disposition = str(row.get("final_disposition", "")).strip()
        final_disposition_lower = final_disposition.lower()
        formula_status = row.get("formula_status", "")
        if formula_status not in CASE_VALIDATION_FORMULA_STATUS:
            errors.append(
                f"{case_id}: formula_status {formula_status!r} is not submission-ready"
            )
        if not str(row.get("code_status", "")).strip():
            errors.append(f"{case_id}: code_status is empty")
        if row.get("boundary_test_status") != "cross_validation_present":
            errors.append(f"{case_id}: boundary_test_status must be cross_validation_present")
        if not str(row.get("next_action", "")).strip():
            errors.append(f"{case_id}: next_action is empty")

        note_path = row.get("validation_note_path", "")
        has_note = bool(note_path and note_path != "NONE")
        if risk == "P1" and not has_note:
            errors.append(
                f"{case_id}: P1 case-validation risk requires a prior validation-note folder"
            )
        if risk == "P2" and not (
            "plausibility only" in final_disposition_lower
            or "analytic" in final_disposition_lower
            or "control" in final_disposition_lower
        ):
            errors.append(
                f"{case_id}: P2 case-validation risk must be marked plausibility-only "
                "or analytic/control in final_disposition"
            )
        if has_note and risk == "P1" and "accepted" not in final_disposition_lower:
            errors.append(
                f"{case_id}: prior-note P1 row must have an accepted final_disposition"
            )
        if note_path and note_path != "NONE":
            note_dir = ROOT / note_path
            documented_note_dirs.add(note_dir.resolve())
            if not note_dir.exists():
                errors.append(f"{case_id}: validation note path missing: {note_path}")
            else:
                has_docx = any(
                    path.suffix.lower() == ".docx" and not path.name.startswith("~$")
                    for path in note_dir.iterdir()
                )
                if not has_docx:
                    errors.append(f"{case_id}: validation note folder has no real docx: {note_path}")

        test_path = row.get("targeted_physics_test", "")
        if formula_status in {"fixed_targeted_tested", "targeted_tested_no_code_change"}:
            if not test_path or test_path == "NONE":
                errors.append(f"{case_id}: fixed/targeted formula row lacks targeted physics test")
            elif not (ROOT / test_path).exists():
                errors.append(f"{case_id}: targeted physics test missing: {test_path}")
        if (
            risk == "P2"
            and not has_note
            and "plausibility only" in final_disposition_lower
            and test_path
            and test_path != "NONE"
            and (ROOT / test_path).exists()
        ):
            names = _case_physics_test_function_names(ROOT / test_path)
            physics_names = [
                name
                for name in names
                if not any(
                    fragment in name
                    for fragment in GENERIC_CASE_PHYSICS_TEST_NAME_FRAGMENTS
                )
            ]
            if len(names) < MIN_P2_PLAUSIBILITY_TEST_FUNCTIONS:
                errors.append(
                    f"{case_id}: P2 plausibility-only targeted physics test "
                    f"has only {len(names)} test function(s); expected at least "
                    f"{MIN_P2_PLAUSIBILITY_TEST_FUNCTIONS}"
                )
            if len(physics_names) < MIN_P2_PLAUSIBILITY_PHYSICS_TEST_FUNCTIONS:
                errors.append(
                    f"{case_id}: P2 plausibility-only targeted physics test "
                    "needs at least "
                    f"{MIN_P2_PLAUSIBILITY_PHYSICS_TEST_FUNCTIONS} non-generic "
                    "physics trend/boundary tests"
                )

        cross_validation = ROOT / "tests" / "_cross_validation" / f"{case_id}.py"
        if case_id and not cross_validation.exists():
            errors.append(f"{case_id}: cross-validation module missing: {cross_validation}")

    if validation_root.exists():
        note_dirs = {
            path.parent.resolve()
            for path in validation_root.rglob("*.docx")
            if not path.name.startswith("~$")
        }
        omitted = sorted(str(path.relative_to(ROOT)) for path in note_dirs - documented_note_dirs)
        if omitted:
            errors.append(f"case validation status omits prior validation note folders: {omitted}")
    return errors


def check_case_validation_sources(status_csv: Path, sources_csv: Path) -> list[str]:
    """Ensure the case audit ledger is pinned to the current case source files."""
    errors: list[str] = []
    if not sources_csv.exists():
        return [f"case validation source fingerprint CSV does not exist: {sources_csv}"]

    try:
        with status_csv.open("r", encoding="utf-8", newline="") as f:
            status_rows = list(csv.DictReader(f))
        with sources_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            source_columns = list(reader.fieldnames or [])
            source_rows = list(reader)
    except Exception as exc:
        return [f"could not parse case validation source fingerprints: {type(exc).__name__}: {exc}"]

    if source_columns != CASE_VALIDATION_SOURCE_COLUMNS:
        errors.append(
            "case validation source schema mismatch: "
            f"expected {CASE_VALIDATION_SOURCE_COLUMNS}, observed {source_columns}"
        )

    status_cases = {row.get("case_id", "") for row in status_rows if row.get("case_id")}
    source_cases = [row.get("case_id", "") for row in source_rows]
    source_set = {case_id for case_id in source_cases if case_id}
    duplicates = sorted({case_id for case_id in source_cases if source_cases.count(case_id) > 1})
    if source_set != status_cases:
        errors.append(
            "case validation source coverage mismatch: "
            f"missing={sorted(status_cases - source_set)}, extra={sorted(source_set - status_cases)}"
        )
    if duplicates:
        errors.append(f"case validation source duplicates cases: {duplicates}")

    by_case = {row.get("case_id", ""): row for row in source_rows}
    for case_id in sorted(status_cases):
        row = by_case.get(case_id)
        if row is None:
            continue
        source, observed_sha, error = _current_case_source_hash(case_id)
        if error:
            errors.append(f"{case_id}: current case source unresolved: {error}")
            continue
        observed_source = _repo_relative_path(source) if source else "<unknown>"
        expected_source = str(row.get("case_source_path", "")).replace("\\", "/")
        if expected_source != observed_source:
            errors.append(
                f"{case_id}: case validation source path mismatch: "
                f"expected {observed_source}, observed {expected_source!r}"
            )
        expected_sha = row.get("case_source_sha256")
        if expected_sha != observed_sha:
            errors.append(
                f"{case_id}: case validation source hash mismatch for {observed_source}: "
                f"expected {observed_sha}, observed {expected_sha}"
            )
    return errors


def _case_validation_note_source_rows() -> list[dict[str, str]]:
    generator_path = ROOT / "scripts" / "update_case_validation_note_sources.py"
    spec = importlib.util.spec_from_file_location(
        "update_case_validation_note_sources_for_readiness",
        generator_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {generator_path}")
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    return generator.current_rows()


def _case_physics_test_function_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return sorted(
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    )


def check_case_validation_note_sources(note_sources_csv: Path) -> list[str]:
    """Ensure the human validation notes and references are hash-pinned."""
    errors: list[str] = []
    if not note_sources_csv.exists():
        return [
            f"case validation note-source fingerprint CSV does not exist: {note_sources_csv}"
        ]

    try:
        with note_sources_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            columns = list(reader.fieldnames or [])
            rows = list(reader)
    except Exception as exc:
        return [
            f"could not parse case validation note-source fingerprints: "
            f"{type(exc).__name__}: {exc}"
        ]

    if columns != CASE_VALIDATION_NOTE_SOURCE_COLUMNS:
        errors.append(
            "case validation note-source schema mismatch: "
            f"expected {CASE_VALIDATION_NOTE_SOURCE_COLUMNS}, observed {columns}"
        )

    expected_rows = _case_validation_note_source_rows()
    expected_by_path = {row["path"]: row for row in expected_rows}
    observed_paths = [row.get("path", "") for row in rows]
    observed_by_path = {path: row for path, row in zip(observed_paths, rows) if path}
    duplicates = sorted({path for path in observed_paths if observed_paths.count(path) > 1})
    missing = sorted(set(expected_by_path) - set(observed_by_path))
    extra = sorted(set(observed_by_path) - set(expected_by_path))
    if missing or extra:
        errors.append(
            "case validation note-source coverage mismatch: "
            f"missing={missing}, extra={extra}"
        )
    if duplicates:
        errors.append(f"case validation note-source duplicates paths: {duplicates}")

    for path, expected in expected_by_path.items():
        observed = observed_by_path.get(path)
        if observed is None:
            continue
        if str(observed.get("size_bytes", "")) != expected["size_bytes"]:
            errors.append(
                f"{path}: case validation note-source size mismatch: "
                f"expected {expected['size_bytes']}, observed {observed.get('size_bytes')}"
            )
        if observed.get("sha256") != expected["sha256"]:
            errors.append(
                f"{path}: case validation note-source hash mismatch: "
                f"expected {expected['sha256']}, observed {observed.get('sha256')}"
            )
    return errors


def check_case_validation_summary(
    status_csv: Path,
    summary_md: Path,
    note_sources_csv: Path = CASE_VALIDATION_NOTE_SOURCES_DEFAULT,
) -> list[str]:
    """Ensure the generated case-validation summary is present and fresh."""
    if not summary_md.exists():
        return [f"case validation summary does not exist: {summary_md}"]
    try:
        generator_path = ROOT / "scripts" / "generate_case_validation_summary.py"
        spec = importlib.util.spec_from_file_location(
            "generate_case_validation_summary_for_readiness",
            generator_path,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load {generator_path}")
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)

        rows = generator._read_rows(status_csv.resolve())
        strict_errors = generator.validation_errors(rows)
        if strict_errors:
            return [
                "case validation summary strict evidence-chain errors: "
                + "; ".join(strict_errors)
            ]
        note_sources = note_sources_csv.resolve()
        note_source_rows = (
            len(generator._read_rows(note_sources))
            if note_sources.exists()
            else None
        )
        expected = generator.render_markdown(
            rows,
            note_source_rows=note_source_rows,
        )
        observed = summary_md.read_text(encoding="utf-8")
    except Exception as exc:
        return [f"could not validate case validation summary: {type(exc).__name__}: {exc}"]
    if observed != expected:
        return [
            "case validation summary is stale; regenerate with "
            "python scripts/generate_case_validation_summary.py"
        ]
    return []


def check_case_experiment_readiness(
    readiness_csv: Path = CASE_EXPERIMENT_READINESS_DEFAULT,
) -> list[str]:
    """Ensure the case experiment-readiness ledger is current and clean."""
    try:
        from scripts.audit_publication_readiness import summarize_case_readiness
    except Exception as exc:
        return [f"could not import case experiment readiness gate: {type(exc).__name__}: {exc}"]

    gate = summarize_case_readiness(readiness_csv)
    if gate.status == "PASS":
        return []
    return [
        "case experiment readiness is not submission-ready: "
        f"{gate.status}; {gate.evidence} Next action: {gate.next_action}"
    ]


def check_numbers_consistency(numbers_path: Path, manifest_path: Path) -> list[str]:
    """Validate paper/numbers.json against RUN_MANIFEST.json.

    This catches the easy-to-miss failure mode where derived manuscript numbers
    are regenerated from a different method pool, stale partial result, or
    mutated report while the manifest still describes the archived sweep.
    """
    errors: list[str] = []
    if not numbers_path.exists():
        return [f"numbers.json does not exist: {numbers_path}"]
    if not manifest_path.exists():
        return [f"RUN_MANIFEST.json does not exist: {manifest_path}"]

    try:
        numbers = json.loads(numbers_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"could not parse numbers.json: {type(exc).__name__}: {exc}"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"could not parse RUN_MANIFEST.json: {type(exc).__name__}: {exc}"]

    experiment = manifest.get("experiment", {})
    summary = manifest.get("results_summary", {})
    status_counts = summary.get("status_counts", {}) if isinstance(summary, dict) else {}
    budgets = experiment.get("budgets", [])
    n_cases = experiment.get("n_cases_declared")
    n_methods = experiment.get("n_methods_declared")
    n_repeats = experiment.get("n_repeats")
    expected_total = None
    if all(isinstance(v, int) for v in [n_cases, n_methods, n_repeats]) and isinstance(budgets, list):
        expected_total = int(n_cases) * int(n_methods) * len(budgets) * int(n_repeats)

    checks = {
        "nCases": n_cases,
        "nMethods": n_methods,
        "nRepeats": n_repeats,
        "nBudgets": len(budgets) if isinstance(budgets, list) else None,
        "seedStart": experiment.get("seed_start"),
        "totalCells": expected_total,
        "totalRows": summary.get("n_rows"),
        "okRows": status_counts.get("ok"),
        "timeoutRows": status_counts.get("timeout", 0),
        "failedRows": status_counts.get("failed", 0),
        "nCasesCovered": summary.get("n_cases_observed"),
        "nMethodsCovered": summary.get("n_methods_observed"),
    }
    for key, expected in checks.items():
        if expected is None:
            continue
        observed = numbers.get(key)
        if observed != expected:
            errors.append(
                f"numbers.json {key} mismatch: expected {expected!r} from RUN_MANIFEST, "
                f"observed {observed!r}"
            )

    budget_list = ", ".join(str(b) for b in budgets) if isinstance(budgets, list) else None
    if budget_list is not None and numbers.get("budgetList") != budget_list:
        errors.append(
            f"numbers.json budgetList mismatch: expected {budget_list!r}, "
            f"observed {numbers.get('budgetList')!r}"
        )

    git_commit = manifest.get("git", {}).get("commit")
    if git_commit:
        expected_short = str(git_commit)[:7]
        if numbers.get("gitCommit") != expected_short:
            errors.append(
                f"numbers.json gitCommit mismatch: expected {expected_short!r}, "
                f"observed {numbers.get('gitCommit')!r}"
            )
    return errors


def check_numbers_submission_completeness(numbers_path: Path) -> list[str]:
    """Reject submission-critical numbers that still carry draft sentinels."""
    errors: list[str] = []
    if not numbers_path.exists():
        return [f"numbers.json does not exist: {numbers_path}"]
    try:
        numbers = json.loads(numbers_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"could not parse numbers.json: {type(exc).__name__}: {exc}"]

    for key in sorted(NUMBERS_SUBMISSION_REQUIRED_NON_NA):
        if key not in numbers:
            errors.append(f"numbers.json missing submission-critical key {key}")
            continue
        value = numbers.get(key)
        if value is None:
            errors.append(f"numbers.json {key} is null; regenerate required paper artifacts")
            continue
        if isinstance(value, str) and value.strip().upper() in {"", "N/A", "NA", "NAN"}:
            errors.append(
                f"numbers.json {key} is {value!r}; regenerate required paper artifacts "
                "before submission"
            )
    return errors


def check_numbers_provenance(numbers_path: Path, paper_dir: Path) -> list[str]:
    """Validate paper/numbers.json source-input fingerprints."""
    errors: list[str] = []
    if not numbers_path.exists():
        return [f"numbers.json does not exist: {numbers_path}"]
    try:
        numbers = json.loads(numbers_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"could not parse numbers.json: {type(exc).__name__}: {exc}"]

    provenance = numbers.get("_provenance")
    if not isinstance(provenance, dict):
        return ["numbers.json missing _provenance block; rerun fill_paper_numbers.py"]
    if provenance.get("script") != "scripts/fill_paper_numbers.py":
        errors.append(
            "numbers.json _provenance.script mismatch: "
            f"observed {provenance.get('script')!r}"
        )
    inputs = provenance.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        errors.append("numbers.json _provenance has no inputs entries")
        return errors

    labels: list[str] = []
    for i, item in enumerate(inputs):
        if not isinstance(item, dict):
            errors.append(f"numbers.json _provenance input #{i}: malformed input entry {item!r}")
            continue
        raw_label = item.get("label")
        if not isinstance(raw_label, str) or not raw_label.strip():
            errors.append(f"numbers.json _provenance input #{i}: missing non-empty label")
            continue
        label = raw_label.strip()
        labels.append(label)
        if item.get("exists") is not True:
            errors.append(
                f"numbers.json _provenance input {label}: exists must be true "
                "for submission-critical source inputs"
            )
        expected_sha = item.get("sha256")
        if not isinstance(expected_sha, str) or not _SHA256_RE.fullmatch(expected_sha.strip()):
            errors.append(
                f"numbers.json _provenance input {label}: missing or invalid sha256"
            )

    label_counts = {label: labels.count(label) for label in set(labels)}
    duplicate_labels = sorted(label for label, count in label_counts.items() if count > 1)
    if duplicate_labels:
        errors.append(
            "numbers.json _provenance has duplicate input labels: "
            f"{duplicate_labels}"
        )

    for label in sorted(NUMBERS_PROVENANCE_REQUIRED_LABELS - set(labels)):
        errors.append(f"numbers.json _provenance missing required input label {label}")
    for item in inputs:
        label = item.get("label", "input") if isinstance(item, dict) else "input"
        errors.extend(
            _check_manifest_file_entry(
                item,
                paper_dir=paper_dir,
                base_dir=ROOT,
                context=f"numbers.json _provenance input {label}",
            )
        )
    return errors


def check_prereg_manifest_config(manifest_path: Path) -> list[str]:
    """Validate RUN_MANIFEST experiment knobs against the headline protocol."""
    errors: list[str] = []
    if not manifest_path.exists():
        return [f"RUN_MANIFEST.json does not exist: {manifest_path}"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"could not parse RUN_MANIFEST.json: {type(exc).__name__}: {exc}"]

    experiment = manifest.get("experiment", {})
    checks = {
        "pool": PREREG_POOL,
        "budgets": PREREG_BUDGETS,
        "n_repeats": PREREG_N_REPEATS,
        "n_test": PREREG_N_TEST,
        "seed_strategy": PREREG_SEED_STRATEGY,
        "seed_start": PREREG_SEED_START,
        "n_cases_declared": PREREG_N_CASES,
        "n_methods_declared": PREREG_N_METHODS,
    }
    for key, expected in checks.items():
        observed = experiment.get(key)
        if observed != expected:
            errors.append(
                f"RUN_MANIFEST experiment.{key} deviates from preregistration: "
                f"expected {expected!r}, observed {observed!r}"
            )

    cases = experiment.get("case_ids_declared")
    if isinstance(cases, list) and len(cases) != PREREG_N_CASES:
        errors.append(
            f"RUN_MANIFEST case_ids_declared count deviates from preregistration: "
            f"expected {PREREG_N_CASES}, observed {len(cases)}"
        )
    methods = experiment.get("methods_declared")
    if not isinstance(methods, list):
        errors.append("RUN_MANIFEST experiment.methods_declared is missing or malformed")
    else:
        observed_methods = [str(method) for method in methods]
        if len(observed_methods) != PREREG_N_METHODS:
            errors.append(
                f"RUN_MANIFEST methods_declared count deviates from preregistration: "
                f"expected {PREREG_N_METHODS}, observed {len(observed_methods)}"
            )
        try:
            from benchmark.methods import nature_pool

            expected_methods = sorted(method.name for method in nature_pool())
        except Exception as exc:
            errors.append(
                "current nature_pool methods could not be resolved for "
                f"preregistration check: {type(exc).__name__}: {exc}"
            )
        else:
            missing = sorted(set(expected_methods) - set(observed_methods))
            extra = sorted(set(observed_methods) - set(expected_methods))
            if missing or extra:
                errors.append(
                    "RUN_MANIFEST methods_declared deviates from current "
                    f"nature_pool: missing={missing}, extra={extra}"
                )
    return errors


def _parse_number_macros(numbers_tex_path: Path) -> tuple[dict[str, str], list[str]]:
    text = numbers_tex_path.read_text(encoding="utf-8")
    macros: dict[str, str] = {}
    seen: set[str] = set()
    duplicates: list[str] = []
    for name, value in _NEWCOMMAND_RE.findall(text):
        macro = f"\\{name}"
        if macro in seen and macro not in duplicates:
            duplicates.append(macro)
        seen.add(macro)
        macros[macro] = value
    return macros, sorted(duplicates)


def _normalise_latex_macro_value(value: str) -> str:
    """Convert the small set of LaTeX escapes emitted by the number writer."""
    replacements = {
        r"\%": "%",
        r"\_": "_",
        r"\#": "#",
        r"\&": "&",
        r"\$": "$",
        r"\{": "{",
        r"\}": "}",
    }
    for escaped, plain in replacements.items():
        value = value.replace(escaped, plain)
    return value


def _strip_thousands_separator(value: str) -> str:
    """``81,600`` -> ``81600``; anything else is returned unchanged.

    The typeset form of a large integer may carry a thousands separator while
    ``numbers.json`` keeps the plain integer that the manifest gate compares
    against.  Both spellings denote the same number, so the consistency check
    must accept either rather than flagging a formatting choice as a mismatch.
    """
    text = value.strip()
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+", text):
        return text.replace(",", "")
    return text


def check_numbers_tex_consistency(numbers_tex_path: Path, numbers_path: Path) -> list[str]:
    """Validate paper/numbers.tex macro values against paper/numbers.json."""
    errors: list[str] = []
    if not numbers_tex_path.exists():
        return [f"numbers.tex does not exist: {numbers_tex_path}"]
    if not numbers_path.exists():
        return [f"numbers.json does not exist: {numbers_path}"]

    try:
        numbers = json.loads(numbers_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"could not parse numbers.json: {type(exc).__name__}: {exc}"]
    try:
        macros, duplicate_macros = _parse_number_macros(numbers_tex_path)
    except Exception as exc:
        return [f"could not parse numbers.tex: {type(exc).__name__}: {exc}"]
    for macro in duplicate_macros:
        errors.append(
            f"numbers.tex defines macro {macro} more than once; "
            "regenerate paper/numbers.tex and paper/numbers.json together"
        )

    for key, macro_names in NUMBER_MACROS.items():
        if key not in numbers:
            continue
        expected = str(numbers[key])
        for macro in macro_names:
            observed = macros.get(macro)
            if observed is None:
                errors.append(f"numbers.tex missing macro {macro} for numbers.json key {key}")
            elif (
                _normalise_latex_macro_value(observed) != expected
                and _strip_thousands_separator(_normalise_latex_macro_value(observed))
                != _strip_thousands_separator(expected)
            ):
                errors.append(
                    f"numbers.tex macro {macro} mismatch for {key}: "
                    f"expected {expected!r}, observed {observed!r}"
                )
    return errors


FIGURE_ARTIFACT_SUBDIRS = ("pdf", "svg", "png", "tiff")


def _portable_path(raw_path: str, paper_dir: Path) -> Path:
    p = Path(raw_path)
    if p.exists():
        return p
    if p.name.startswith("fig") and p.suffix in {".pdf", ".svg", ".png", ".tiff"}:
        # Every figure artefact lives under a format-named subdirectory, so
        # resolve by the declared parent when it names a known format and fall
        # back to matching the suffix otherwise.  Restricting this to pdf/png
        # silently resolved svg and tiff to paper/figN.<ext>.
        if p.parent.name in FIGURE_ARTIFACT_SUBDIRS:
            return paper_dir / "figures" / p.parent.name / p.name
        matches = [
            paper_dir / "figures" / sub / p.name
            for sub in FIGURE_ARTIFACT_SUBDIRS
            if (paper_dir / "figures" / sub / p.name).exists()
        ]
        if len(matches) == 1:
            return matches[0]
        return paper_dir / "figures" / p.name
    if p.suffix == ".csv":
        return paper_dir / "figdata" / p.name
    return paper_dir / p.name


def _resolve_repo_or_paper_path(raw_path: str, paper_dir: Path) -> Path:
    p = Path(raw_path)
    if p.exists():
        return p
    if p.is_absolute():
        return p
    paper_path = paper_dir / p
    if paper_path.exists():
        return paper_path
    repo_path = ROOT / p
    if repo_path.exists():
        return repo_path
    return _portable_path(raw_path, paper_dir)


def _check_manifest_file_entry(
    item: object,
    *,
    paper_dir: Path,
    base_dir: Path | None = None,
    context: str,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(item, dict):
        return [f"{context}: malformed file entry {item!r}"]
    raw = item.get("path")
    if not isinstance(raw, str) or not raw.strip():
        return [f"{context}: file entry missing path"]
    if item.get("exists") is False:
        errors.append(f"{context}: recorded missing file {raw}")
    if base_dir is not None and not Path(raw).is_absolute():
        base_path = base_dir / raw
        path = base_path if base_path.exists() else _resolve_repo_or_paper_path(raw, paper_dir)
    else:
        path = _resolve_repo_or_paper_path(raw, paper_dir)
    if not path.exists():
        errors.append(f"{context}: missing file {path}")
        return errors
    expected_size = item.get("size_bytes")
    if expected_size is not None:
        try:
            expected_size_int = int(expected_size)
        except (TypeError, ValueError):
            errors.append(f"{context}: size_bytes is not an integer for {raw}")
        else:
            observed = path.stat().st_size
            if observed != expected_size_int:
                errors.append(
                    f"{context}: size mismatch for {path}: "
                    f"expected {expected_size_int}, observed {observed}"
                )
    expected_sha = item.get("sha256")
    if expected_sha:
        if _sha256(path).lower() != str(expected_sha).lower():
            errors.append(f"{context}: sha256 mismatch for {path}")
    return errors


def _resolve_manifest_file_entry_path(
    item: object,
    *,
    paper_dir: Path,
    base_dir: Path | None = None,
) -> Path | None:
    if not isinstance(item, dict):
        return None
    raw = item.get("path")
    if not isinstance(raw, str) or not raw.strip():
        return None
    if base_dir is not None and not Path(raw).is_absolute():
        base_path = base_dir / raw
        if base_path.exists():
            return base_path
    return _resolve_repo_or_paper_path(raw, paper_dir)


def _figdata_results_csv_input_path(
    manifest: dict[str, object],
    *,
    paper_dir: Path,
    base_dir: Path,
) -> Path | None:
    inputs = manifest.get("inputs")
    if not isinstance(inputs, list):
        return None
    for item in inputs:
        if not isinstance(item, dict):
            continue
        if str(item.get("label", "")).strip() == "results_csv":
            return _resolve_manifest_file_entry_path(
                item,
                paper_dir=paper_dir,
                base_dir=base_dir,
            )
    return None


def _current_figdata_results_summary(results_csv: Path) -> dict[str, object]:
    import pandas as pd

    df = pd.read_csv(results_csv, low_memory=False)
    summary: dict[str, object] = {
        "exists": True,
        "n_rows": int(len(df)),
    }
    if "case_id" in df.columns:
        summary["n_cases"] = int(df["case_id"].dropna().astype(str).nunique())
    if "method" in df.columns:
        summary["n_methods"] = int(df["method"].dropna().astype(str).nunique())
    for column, key in (("budget_mult", "budgets"), ("repeat", "repeats")):
        if column not in df.columns:
            continue
        values = []
        for value in df[column].dropna().unique():
            try:
                parsed = int(float(str(value)))
            except ValueError:
                continue
            values.append(parsed)
        summary[key] = sorted(set(values))
    for column, key in (
        ("status", "status_counts"),
        ("failure_category", "failure_category_counts"),
    ):
        if column in df.columns:
            summary[key] = {
                str(label): int(count)
                for label, count in (
                    df[column]
                    .fillna("")
                    .astype(str)
                    .value_counts()
                    .sort_index()
                    .items()
                )
            }
    return summary


def _check_figdata_results_summary(
    manifest: dict[str, object],
    *,
    results_csv_path: Path | None = None,
) -> list[str]:
    errors: list[str] = []
    summary = manifest.get("results_summary")
    if not isinstance(summary, dict):
        return [
            "figdata_manifest.json missing results_summary; regenerate with "
            "scripts/prepare_figdata.py so figure-data provenance records the "
            "headline CSV row/status/failure-category counts"
        ]
    n_rows = summary.get("n_rows")
    try:
        n_rows_int = int(n_rows)
    except (TypeError, ValueError):
        errors.append("figdata_manifest.json results_summary.n_rows must be an integer")
        n_rows_int = None
    else:
        if n_rows_int < 0:
            errors.append("figdata_manifest.json results_summary.n_rows must be non-negative")

    for key in ("status_counts", "failure_category_counts"):
        counts = summary.get(key)
        if not isinstance(counts, dict) or not counts:
            errors.append(f"figdata_manifest.json results_summary.{key} must be a non-empty object")
            continue
        total = 0
        for label, value in counts.items():
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                errors.append(
                    f"figdata_manifest.json results_summary.{key}.{label} must be an integer"
                )
                continue
            if parsed < 0:
                errors.append(
                    f"figdata_manifest.json results_summary.{key}.{label} must be non-negative"
                )
            total += parsed
        if n_rows_int is not None and total != n_rows_int:
            errors.append(
                f"figdata_manifest.json results_summary.{key} total {total} "
                f"does not match n_rows {n_rows_int}"
            )
    if results_csv_path is not None and results_csv_path.exists() and results_csv_path.is_file():
        try:
            observed = _current_figdata_results_summary(results_csv_path)
        except Exception as exc:
            errors.append(
                "figdata_manifest.json results_summary could not be checked "
                f"against current results_csv: {type(exc).__name__}: {exc}"
            )
        else:
            for key, observed_value in observed.items():
                expected_value = summary.get(key)
                if expected_value != observed_value:
                    errors.append(
                        "figdata_manifest.json results_summary "
                        f"{key} mismatch against current results_csv: "
                        f"expected {observed_value!r}, observed {expected_value!r}"
                    )
    return errors


def check_figdata_manifest(figdata_dir: Path, paper_dir: Path) -> list[str]:
    """Validate figdata provenance against source inputs and derived CSVs."""
    errors: list[str] = []
    manifest_path = figdata_dir / "figdata_manifest.json"
    if not manifest_path.exists():
        return [f"figdata_manifest.json does not exist: {manifest_path}"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"could not parse figdata_manifest.json: {type(exc).__name__}: {exc}"]

    manifest_errors = manifest.get("errors", [])
    if manifest_errors:
        errors.append(f"figdata_manifest.json contains generation errors: {manifest_errors!r}")
    status = str(manifest.get("status", "")).strip()
    if status != "complete":
        errors.append(
            f"figdata_manifest.json status is {status!r}; regenerate complete figdata package"
        )

    policy = manifest.get("analysis_policy")
    if not isinstance(policy, dict):
        errors.append(
            "figdata_manifest.json missing analysis_policy; regenerate with "
            "scripts/prepare_figdata.py so D1/D2/D3/Fig. 6 primary policies "
            "are machine-checkable"
        )
    else:
        for key, expected in EXPECTED_FIGDATA_ANALYSIS_POLICY.items():
            observed = policy.get(key)
            if observed != expected:
                errors.append(
                    "figdata_manifest.json analysis_policy "
                    f"{key} mismatch: expected {expected!r}, observed {observed!r}"
                )

    results_csv_path = _figdata_results_csv_input_path(
        manifest,
        paper_dir=paper_dir,
        base_dir=figdata_dir.parent,
    )
    errors.extend(
        _check_figdata_results_summary(
            manifest,
            results_csv_path=results_csv_path,
        )
    )

    inputs = manifest.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        errors.append("figdata_manifest.json has no inputs entries")
    else:
        labels = {
            str(item.get("label", ""))
            for item in inputs
            if isinstance(item, dict)
        }
        required_input_labels = {
            "results_csv",
            "d1_correlation",
            "d1_rank_comparison",
            "d2_learning_curves",
            "d2_rule_summary",
            "d3_reliability_per_method",
        }
        for required in required_input_labels:
            if required not in labels:
                errors.append(f"figdata_manifest.json missing required input label {required}")
        d1_modes = set(
            ANALYSIS_ARTIFACT_SCHEMAS["d1_correlation_json"].get("required_pairing_modes", set())
        )
        rank_inputs = [
            item
            for item in inputs
            if isinstance(item, dict) and str(item.get("label", "")) == "d1_rank_comparison"
        ]
        rank_modes = set()
        for item in rank_inputs:
            raw_path = str(item.get("path", ""))
            for part in Path(raw_path).parts:
                if part.startswith("pairing_"):
                    rank_modes.add(part.removeprefix("pairing_"))
        missing_rank_modes = sorted(d1_modes - rank_modes)
        if missing_rank_modes:
            errors.append(
                "figdata_manifest.json missing d1_rank_comparison inputs for "
                f"pairing modes {missing_rank_modes}"
            )
        for item in inputs:
            label = item.get("label", "input") if isinstance(item, dict) else "input"
            errors.extend(
                _check_manifest_file_entry(
                    item,
                    paper_dir=paper_dir,
                    base_dir=figdata_dir.parent,
                    context=f"figdata_manifest input {label}",
                )
            )

    figdata_files = manifest.get("figdata_files")
    if not isinstance(figdata_files, list) or not figdata_files:
        errors.append("figdata_manifest.json has no figdata_files entries")
    else:
        seen_paths = set()
        for item in figdata_files:
            if isinstance(item, dict) and isinstance(item.get("path"), str):
                seen_paths.add(Path(str(item["path"])).name)
            errors.extend(
                _check_manifest_file_entry(
                    item,
                    paper_dir=paper_dir,
                    base_dir=figdata_dir.parent,
                    context="figdata_manifest figdata_files",
                )
            )
        missing = sorted(set(FIGDATA_SCHEMAS) - seen_paths)
        if missing:
            errors.append(f"figdata_manifest.json missing required figdata CSV entries: {missing}")
    return errors


def check_figure_manifest(
    figure_manifest_path: Path,
    paper_dir: Path,
    *,
    allow_draft_figdata: bool = False,
) -> list[str]:
    """Validate paper/figure_manifest.json and all referenced hashes."""
    errors: list[str] = []
    if not figure_manifest_path.exists():
        return [f"figure_manifest.json does not exist: {figure_manifest_path}"]

    try:
        manifest = json.loads(figure_manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"could not parse figure_manifest.json: {type(exc).__name__}: {exc}"]

    status = manifest.get("status")
    if status not in (None, "complete"):
        errors.append(
            "figure_manifest.json status is "
            f"{status!r}; rerun plots_nature.py for the complete figure package"
        )

    manifest_errors = manifest.get("errors", [])
    if manifest_errors:
        errors.append(f"figure_manifest.json contains generation errors: {manifest_errors!r}")

    figdata_manifest = manifest.get("figdata_manifest")
    figure_policy = manifest.get("analysis_policy")
    linked_figdata_status: str | None = None
    if not isinstance(figdata_manifest, dict):
        errors.append("figure_manifest.json missing figdata_manifest entry")
    else:
        raw = figdata_manifest.get("path")
        expected_sha = figdata_manifest.get("sha256")
        if not isinstance(raw, str) or not raw.strip():
            errors.append("figure_manifest.json figdata_manifest entry missing path")
        else:
            path = _resolve_repo_or_paper_path(raw, paper_dir)
            if not path.exists():
                errors.append(f"figure_manifest.json missing figdata_manifest artifact: {path}")
            elif expected_sha and _sha256(path).lower() != str(expected_sha).lower():
                errors.append(f"figure_manifest.json figdata_manifest sha256 mismatch: {path}")
            elif path.exists():
                try:
                    linked_figdata = json.loads(path.read_text(encoding="utf-8"))
                except Exception as exc:
                    errors.append(
                        "figure_manifest.json linked figdata_manifest is not parseable: "
                        f"{type(exc).__name__}: {exc}"
                    )
                else:
                    linked_errors = linked_figdata.get("errors", [])
                    linked_status = str(linked_figdata.get("status", "")).strip()
                    linked_figdata_status = linked_status
                    linked_policy = linked_figdata.get("analysis_policy")
                    draft_figdata_allowed = (
                        allow_draft_figdata
                        and linked_status == "draft_pending_full_results"
                    )
                    if linked_errors and not draft_figdata_allowed:
                        errors.append(
                            "figure_manifest.json linked figdata_manifest contains "
                            f"generation errors: {linked_errors!r}"
                        )
                    if linked_status != "complete" and not draft_figdata_allowed:
                        errors.append(
                            "figure_manifest.json linked figdata_manifest status is "
                            f"{linked_status!r}; regenerate complete figdata before figures"
                        )
                    if linked_status == "complete":
                        if not isinstance(linked_policy, dict):
                            errors.append(
                                "figure_manifest.json linked figdata_manifest missing "
                                "analysis_policy; regenerate figdata before figures"
                            )
                        elif not isinstance(figure_policy, dict):
                            errors.append(
                                "figure_manifest.json missing analysis_policy copied from "
                                "figdata_manifest.json; rerun plots_nature.py"
                            )
                        elif figure_policy != linked_policy:
                            errors.append(
                                "figure_manifest.json analysis_policy does not match "
                                "linked figdata_manifest.json analysis_policy"
                            )

    draft_policy_allowed = allow_draft_figdata and linked_figdata_status == "draft_pending_full_results"
    if isinstance(figure_policy, dict):
        for key, expected in EXPECTED_FIGDATA_ANALYSIS_POLICY.items():
            observed = figure_policy.get(key)
            if observed != expected:
                errors.append(
                    "figure_manifest.json analysis_policy "
                    f"{key} mismatch: expected {expected!r}, observed {observed!r}"
                )
    elif not draft_policy_allowed and linked_figdata_status != "complete":
        errors.append(
            "figure_manifest.json missing analysis_policy; rerun plots_nature.py "
            "after complete figdata generation so figure artifacts are bound to "
            "the preregistered D1/D2/D3/Fig. 6 analysis policy"
        )

    figures = manifest.get("figures")
    if not isinstance(figures, list) or not figures:
        return errors + ["figure_manifest.json has no figures entries"]

    seen = set()
    for item in figures:
        figure = item.get("figure") if isinstance(item, dict) else None
        if not figure:
            errors.append(f"figure manifest entry missing figure id: {item!r}")
            continue
        seen.add(str(figure))
        # pdf and png are mandatory; svg and tiff are verified whenever the
        # manifest declares them, so an export format that is present in the
        # package cannot silently drift or go missing.
        for key in ("pdf", "png"):
            if not item.get(key):
                errors.append(f"{figure}: missing {key} path")
        for key in ("pdf", "svg", "png", "tiff"):
            raw = item.get(key)
            expected_sha = item.get(f"{key}_sha256")
            if not raw:
                continue
            path = _portable_path(str(raw), paper_dir)
            if not path.exists():
                errors.append(f"{figure}: missing {key} artifact: {path}")
                continue
            if expected_sha and _sha256(path).lower() != str(expected_sha).lower():
                errors.append(f"{figure}: {key} sha256 mismatch: {path}")
        sources = item.get("source_figdata", [])
        if not isinstance(sources, list) or not sources:
            errors.append(f"{figure}: no source_figdata entries")
            continue
        for source in sources:
            raw = source.get("path") if isinstance(source, dict) else None
            expected_sha = source.get("sha256") if isinstance(source, dict) else None
            if not raw:
                errors.append(f"{figure}: source_figdata entry missing path: {source!r}")
                continue
            path = _portable_path(str(raw), paper_dir)
            if not path.exists():
                errors.append(f"{figure}: missing source figdata: {path}")
                continue
            if expected_sha and _sha256(path).lower() != str(expected_sha).lower():
                errors.append(f"{figure}: source figdata sha256 mismatch: {path}")

    expected = {f"fig{i}" for i in range(1, 7)}
    missing = sorted(expected - seen)
    if missing:
        errors.append(f"figure_manifest.json missing expected figures: {missing}")
    return errors


def check_figure_palette_contract() -> list[str]:
    """Ensure the figure script still uses the CVD-safe palette contract."""
    errors: list[str] = []
    plots_path = ROOT / "scripts" / "plots_nature.py"
    if not plots_path.exists():
        return [f"plots_nature.py does not exist: {plots_path}"]
    try:
        spec = importlib.util.spec_from_file_location(
            "plots_nature_for_submission_ready",
            plots_path,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load {plots_path}")
        plots = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plots)
    except Exception as exc:
        return [f"could not validate plots_nature palette: {type(exc).__name__}: {exc}"]

    wong = tuple(getattr(plots, "WONG_PALETTE", ()))
    if len(wong) != 8 or len(set(wong)) != 8 or any(
        not (isinstance(color, str) and re.fullmatch(r"#[0-9A-Fa-f]{6}", color))
        for color in wong
    ):
        errors.append("plots_nature.py WONG_PALETTE must contain 8 distinct hex colours")

    palette_b = tuple(getattr(plots, "PALETTE_B", ()))
    if len(palette_b) != 4 or len(set(palette_b)) != 4 or any(color not in wong for color in palette_b):
        errors.append("plots_nature.py PALETTE_B must contain 4 distinct WONG_PALETTE colours")

    if str(getattr(plots, "CVD_SEQUENTIAL_CMAP", "")) != "cividis_r":
        errors.append("plots_nature.py CVD_SEQUENTIAL_CMAP must be 'cividis_r'")

    source = plots_path.read_text(encoding="utf-8")
    unsafe_tokens = (
        "plt.cm.tab10",
        "plt.cm.tab20",
        "plt.cm.Set1",
        "plt.cm.Paired",
        "plt.cm.RdYlBu",
        "plt.cm.RdYlGn",
        "plt.cm.bwr",
        "plt.cm.coolwarm",
        'cmap="tab10"', 'cmap="tab20"', 'cmap="RdYlBu_r"',
        'cmap="RdYlGn"', 'cmap="bwr"', 'cmap="coolwarm"',
    )
    offenders = [token for token in unsafe_tokens if token in source]
    if offenders:
        errors.append(f"plots_nature.py reintroduced CVD-unsafe colormap(s): {offenders}")

    legacy_hexes = ("#4575b4", "#74add1", "#1a6faf", "#abd9e9")
    legacy = [color for color in legacy_hexes if color in source]
    if legacy:
        errors.append(f"plots_nature.py contains legacy hard-coded colours: {legacy}")
    return errors


def _is_finite_number(value: str) -> bool:
    if value == "":
        return False
    try:
        return math.isfinite(float(value))
    except ValueError:
        return False


def _is_iso_datetime(value: str) -> bool:
    text = str(value).strip()
    if not text:
        return False
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        datetime.fromisoformat(text)
        return True
    except ValueError:
        return False


def _is_truthy(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _is_falsey(value: str) -> bool:
    return str(value).strip().lower() in {"false", "0", "no", "n", ""}


def _is_bool_text(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y", "false", "0", "no", "n"}


def check_figdata_schema(figdata_dir: Path) -> list[str]:
    """Validate paper/figdata CSVs against the columns consumed by plots."""
    errors: list[str] = []
    if not figdata_dir.exists():
        return [f"figdata directory does not exist: {figdata_dir}"]

    for filename, schema in FIGDATA_SCHEMAS.items():
        path = figdata_dir / filename
        if not path.exists():
            errors.append(f"figdata missing required CSV: {path}")
            continue
        try:
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                columns = set(reader.fieldnames or [])
                rows = list(reader)
        except Exception as exc:
            errors.append(f"could not parse figdata CSV {path}: {type(exc).__name__}: {exc}")
            continue

        required = set(schema.get("required", set())) | FIGDATA_PROVENANCE_COLUMNS
        missing = sorted(required - columns)
        if missing:
            errors.append(f"{filename}: missing required columns {missing}")

        for group in schema.get("one_of", []):
            group_set = set(group)
            if not (group_set & columns):
                errors.append(
                    f"{filename}: missing at least one of columns {sorted(group_set)}"
                )

        if not rows:
            errors.append(f"{filename}: CSV has no data rows")
            continue

        if FIGDATA_PROVENANCE_COLUMNS <= columns:
            empty_sources = [
                i for i, row in enumerate(rows, start=2)
                if not str(row.get("_source", "")).strip()
            ]
            if empty_sources:
                shown = empty_sources[:5]
                more = "" if len(empty_sources) <= 5 else f" (+{len(empty_sources) - 5} more)"
                errors.append(f"{filename}: empty _source values at rows {shown}{more}")

            bad_generated = [
                i for i, row in enumerate(rows, start=2)
                if not _is_iso_datetime(str(row.get("_generated_at", "")))
            ]
            if bad_generated:
                shown = bad_generated[:5]
                more = "" if len(bad_generated) <= 5 else f" (+{len(bad_generated) - 5} more)"
                errors.append(f"{filename}: invalid _generated_at values at rows {shown}{more}")

        numeric_cols = set(schema.get("numeric", set())) & columns
        for col in sorted(numeric_cols):
            bad_rows = [
                i for i, row in enumerate(rows, start=2)
                if not _is_finite_number(str(row.get(col, "")))
            ]
            if bad_rows:
                shown = bad_rows[:5]
                more = "" if len(bad_rows) <= 5 else f" (+{len(bad_rows) - 5} more)"
                errors.append(f"{filename}: non-finite numeric values in {col} at rows {shown}{more}")

        boolean_cols = set(schema.get("boolean", set())) & columns
        for col in sorted(boolean_cols):
            bad_rows = [
                i for i, row in enumerate(rows, start=2)
                if not _is_bool_text(str(row.get(col, "")))
            ]
            if bad_rows:
                shown = bad_rows[:5]
                more = "" if len(bad_rows) <= 5 else f" (+{len(bad_rows) - 5} more)"
                errors.append(f"{filename}: invalid boolean values in {col} at rows {shown}{more}")

        if filename == "fig4_d3_bias.csv" and {"n_runs", "low_n"} <= columns:
            bad_low_n_rows: list[int] = []
            for i, row in enumerate(rows, start=2):
                raw_n = str(row.get("n_runs", ""))
                raw_low_n = str(row.get("low_n", ""))
                if not _is_finite_number(raw_n):
                    continue
                expected_low_n = float(raw_n) < 10
                if expected_low_n and not _is_truthy(raw_low_n):
                    bad_low_n_rows.append(i)
                if not expected_low_n and not _is_falsey(raw_low_n):
                    bad_low_n_rows.append(i)
            if bad_low_n_rows:
                shown = bad_low_n_rows[:5]
                more = "" if len(bad_low_n_rows) <= 5 else f" (+{len(bad_low_n_rows) - 5} more)"
                errors.append(
                    f"{filename}: low_n must equal n_runs < 10 at rows {shown}{more}"
                )

        if filename.startswith("fig2_d1_") and {"analysis_type", "is_primary"} <= columns:
            bad_primary_rows: list[int] = []
            has_expected_primary = False
            for i, row in enumerate(rows, start=2):
                is_primary = _is_truthy(str(row.get("is_primary", "")))
                analysis_type = str(row.get("analysis_type", "")).strip()
                if is_primary:
                    if analysis_type == PRIMARY_D1_ANALYSIS:
                        has_expected_primary = True
                    else:
                        bad_primary_rows.append(i)
            if bad_primary_rows:
                shown = bad_primary_rows[:5]
                more = "" if len(bad_primary_rows) <= 5 else f" (+{len(bad_primary_rows) - 5} more)"
                errors.append(
                    f"{filename}: only {PRIMARY_D1_ANALYSIS!r} rows may have is_primary=True; "
                    f"bad rows {shown}{more}"
                )
            if not has_expected_primary:
                errors.append(
                    f"{filename}: no primary rows for {PRIMARY_D1_ANALYSIS!r}"
                )

        if filename == "fig3_d2_rule.csv" and {"analysis_type", "is_primary"} <= columns:
            bad_primary_rows: list[int] = []
            has_expected_primary = False
            for i, row in enumerate(rows, start=2):
                is_primary = _is_truthy(str(row.get("is_primary", "")))
                analysis_type = str(row.get("analysis_type", "")).strip()
                if is_primary:
                    if analysis_type == PRIMARY_D2_ANALYSIS:
                        has_expected_primary = True
                    else:
                        bad_primary_rows.append(i)
            if bad_primary_rows:
                shown = bad_primary_rows[:5]
                more = "" if len(bad_primary_rows) <= 5 else f" (+{len(bad_primary_rows) - 5} more)"
                errors.append(
                    f"{filename}: only {PRIMARY_D2_ANALYSIS!r} rows may have is_primary=True; "
                    f"bad rows {shown}{more}"
                )
            if not has_expected_primary:
                errors.append(
                    f"{filename}: no primary rows for {PRIMARY_D2_ANALYSIS!r}"
                )

        if filename == "fig6_pareto.csv" and {"analysis_type", "is_primary"} <= columns:
            bad_primary_rows: list[int] = []
            has_expected_primary = False
            for i, row in enumerate(rows, start=2):
                is_primary = _is_truthy(str(row.get("is_primary", "")))
                analysis_type = str(row.get("analysis_type", "")).strip()
                if is_primary:
                    if analysis_type == PRIMARY_FIG6_ANALYSIS:
                        has_expected_primary = True
                    else:
                        bad_primary_rows.append(i)
            if bad_primary_rows:
                shown = bad_primary_rows[:5]
                more = "" if len(bad_primary_rows) <= 5 else f" (+{len(bad_primary_rows) - 5} more)"
                errors.append(
                    f"{filename}: only {PRIMARY_FIG6_ANALYSIS!r} rows may have is_primary=True; "
                    f"bad rows {shown}{more}"
                )
            if not has_expected_primary:
                errors.append(
                    f"{filename}: no primary rows for {PRIMARY_FIG6_ANALYSIS!r}"
                )
    return errors


def _declared_methods_from_manifest(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    raw_methods = manifest.get("experiment", {}).get("methods_declared")
    if not isinstance(raw_methods, list):
        return set()
    methods: set[str] = set()
    for item in raw_methods:
        if item is None:
            continue
        name = str(item).strip()
        if name:
            methods.add(name)
    return methods


def check_figdata_methods(figdata_dir: Path, manifest_path: Path) -> list[str]:
    """Ensure method-bearing figdata was derived from the declared method pool."""
    errors: list[str] = []
    declared = _declared_methods_from_manifest(manifest_path)
    if not declared or not figdata_dir.exists():
        return errors

    for path in sorted(figdata_dir.glob("*.csv")):
        try:
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                if "method" not in set(reader.fieldnames or []):
                    continue
                observed = {
                    str(row.get("method", "")).strip()
                    for row in reader
                    if str(row.get("method", "")).strip()
                }
        except Exception as exc:
            errors.append(f"could not parse figdata methods in {path}: {type(exc).__name__}: {exc}")
            continue

        extra = sorted(observed - declared)
        if extra:
            errors.append(
                f"{path.name}: methods outside RUN_MANIFEST declared pool {extra}"
            )

        if path.name == "fig1_method_catalog.csv":
            missing = sorted(declared - observed)
            if missing:
                errors.append(
                    f"{path.name}: missing RUN_MANIFEST declared methods {missing}"
                )
    return errors


def _check_required_csv_artifact(
    path: Path,
    *,
    required: set[str],
    numeric: set[str] | None = None,
    expected_values: dict[str, set[str]] | None = None,
    required_n_test: int | None = None,
    required_budgets: set[int] | None = None,
    required_pairing_mode: str | None = None,
    required_primary_budgets: set[int] | None = None,
    check_low_n: bool = False,
) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"required analysis artifact missing: {path}"]
    if path.stat().st_size == 0:
        return [f"required analysis artifact is empty: {path}"]
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            columns = set(reader.fieldnames or [])
            rows = list(reader)
    except Exception as exc:
        return [f"could not parse analysis artifact {path}: {type(exc).__name__}: {exc}"]

    missing = sorted(required - columns)
    if missing:
        errors.append(f"{path.name}: missing required columns {missing}")
    if not rows:
        errors.append(f"{path.name}: CSV has no data rows")
        return errors

    for col in sorted((numeric or set()) & columns):
        bad_rows = [
            i for i, row in enumerate(rows, start=2)
            if not _is_finite_number(str(row.get(col, "")))
        ]
        if bad_rows:
            shown = bad_rows[:5]
            more = "" if len(bad_rows) <= 5 else f" (+{len(bad_rows) - 5} more)"
            errors.append(f"{path.name}: non-finite numeric values in {col} at rows {shown}{more}")

    for col, allowed in (expected_values or {}).items():
        if col not in columns:
            continue
        observed = {str(row.get(col, "")).strip() for row in rows}
        bad = sorted(observed - set(allowed))
        if bad:
            errors.append(f"{path.name}: unexpected {col} values {bad}; expected one of {sorted(allowed)}")

    if {"analysis_type", "is_primary"} <= columns and expected_values and "analysis_type" in expected_values:
        primary_rows = [
            row for row in rows
            if _is_truthy(str(row.get("is_primary", "")))
        ]
        allowed = set(expected_values["analysis_type"])
        if not primary_rows:
            errors.append(f"{path.name}: no is_primary=True rows")
        else:
            bad_primary = sorted({
                str(row.get("analysis_type", "")).strip()
                for row in primary_rows
                if str(row.get("analysis_type", "")).strip() not in allowed
            })
            if bad_primary:
                errors.append(
                    f"{path.name}: primary rows have unexpected analysis_type "
                    f"{bad_primary}; expected one of {sorted(allowed)}"
                )

    if required_budgets is not None and "budget_mult" in columns:
        observed_budgets: set[int] = set()
        for row in rows:
            try:
                observed_budgets.add(int(float(str(row.get("budget_mult", "")))))
            except ValueError:
                continue
        missing_budgets = sorted(required_budgets - observed_budgets)
        if missing_budgets:
            errors.append(f"{path.name}: missing budget_mult rows {missing_budgets}")

    if required_pairing_mode is not None and "pairing_mode" in columns:
        observed_modes = {
            str(row.get("pairing_mode", "")).strip()
            for row in rows
            if str(row.get("pairing_mode", "")).strip()
        }
        if observed_modes != {required_pairing_mode}:
            errors.append(
                f"{path.name}: expected pairing_mode={required_pairing_mode!r}, "
                f"observed {sorted(observed_modes)}"
            )

    if required_primary_budgets is not None and {"budget_mult", "is_primary"} <= columns:
        primary_budgets: set[int] = set()
        for row in rows:
            if not _is_truthy(str(row.get("is_primary", ""))):
                continue
            if required_pairing_mode is not None and "pairing_mode" in columns:
                mode = str(row.get("pairing_mode", "")).strip()
                if mode != required_pairing_mode:
                    continue
            try:
                primary_budgets.add(int(float(str(row.get("budget_mult", "")))))
            except ValueError:
                continue
        missing_primary_budgets = sorted(required_primary_budgets - primary_budgets)
        if missing_primary_budgets:
            if required_pairing_mode is not None:
                errors.append(
                    f"{path.name}: missing primary budget_mult rows {missing_primary_budgets} "
                    f"for pairing_mode={required_pairing_mode!r}"
                )
            else:
                errors.append(
                    f"{path.name}: missing primary budget_mult rows {missing_primary_budgets}"
                )

    if check_low_n and {"n_runs", "low_n"} <= columns:
        bad_low_n_rows: list[int] = []
        for i, row in enumerate(rows, start=2):
            raw_n = str(row.get("n_runs", ""))
            raw_low_n = str(row.get("low_n", ""))
            if not _is_finite_number(raw_n):
                continue
            expected_low_n = float(raw_n) < 10
            if expected_low_n and not _is_truthy(raw_low_n):
                bad_low_n_rows.append(i)
            if not expected_low_n and not _is_falsey(raw_low_n):
                bad_low_n_rows.append(i)
        if bad_low_n_rows:
            shown = bad_low_n_rows[:5]
            more = "" if len(bad_low_n_rows) <= 5 else f" (+{len(bad_low_n_rows) - 5} more)"
            errors.append(
                f"{path.name}: low_n must equal n_runs < 10 at rows {shown}{more}"
            )

    if required_n_test is not None and "n_test" in columns:
        has_target = False
        for row in rows:
            try:
                if int(float(str(row.get("n_test", "")))) == required_n_test:
                    has_target = True
                    break
            except ValueError:
                continue
        if not has_target:
            errors.append(f"{path.name}: no n_test={required_n_test} rows for MC-noise submission macro")
    return errors


def _normalise_json_records(payload: object) -> list[dict[str, object]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("records", "correlations", "summary", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _check_required_json_artifact(
    path: Path,
    *,
    required: set[str],
    numeric: set[str] | None = None,
    expected_values: dict[str, set[str]] | None = None,
    required_pairing_modes: set[str] | None = None,
    required_budgets: set[int] | None = None,
    required_primary_pairing_modes: set[str] | None = None,
    required_primary_budgets: set[int] | None = None,
) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"required analysis artifact missing: {path}"]
    if path.stat().st_size == 0:
        return [f"required analysis artifact is empty: {path}"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"could not parse analysis artifact {path}: {type(exc).__name__}: {exc}"]

    records = _normalise_json_records(payload)
    if not records:
        return [f"{path.name}: JSON has no object records"]

    columns = set().union(*(set(row) for row in records))
    missing = sorted(required - columns)
    if missing:
        errors.append(f"{path.name}: missing required keys {missing}")

    for col in sorted((numeric or set()) & columns):
        bad_rows = [
            i for i, row in enumerate(records, start=1)
            if not _is_finite_number(str(row.get(col, "")))
        ]
        if bad_rows:
            shown = bad_rows[:5]
            more = "" if len(bad_rows) <= 5 else f" (+{len(bad_rows) - 5} more)"
            errors.append(f"{path.name}: non-finite numeric values in {col} at records {shown}{more}")

    for col, allowed in (expected_values or {}).items():
        if col not in columns:
            continue
        observed = {str(row.get(col, "")).strip() for row in records}
        bad = sorted(observed - set(allowed))
        if bad:
            errors.append(f"{path.name}: unexpected {col} values {bad}; expected one of {sorted(allowed)}")

    if {"analysis_type", "is_primary"} <= columns and expected_values and "analysis_type" in expected_values:
        primary_rows = [
            row for row in records
            if _is_truthy(str(row.get("is_primary", "")))
        ]
        allowed = set(expected_values["analysis_type"])
        if not primary_rows:
            errors.append(f"{path.name}: no is_primary=True records")
        else:
            bad_primary = sorted({
                str(row.get("analysis_type", "")).strip()
                for row in primary_rows
                if str(row.get("analysis_type", "")).strip() not in allowed
            })
            if bad_primary:
                errors.append(
                    f"{path.name}: primary records have unexpected analysis_type "
                    f"{bad_primary}; expected one of {sorted(allowed)}"
                )

    if required_pairing_modes is not None and "pairing_mode" in columns:
        observed_modes = {
            str(row.get("pairing_mode", "")).strip()
            for row in records
            if str(row.get("pairing_mode", "")).strip()
        }
        missing_modes = sorted(required_pairing_modes - observed_modes)
        if missing_modes:
            errors.append(f"{path.name}: missing pairing_mode records {missing_modes}")

    if required_budgets is not None and "budget_mult" in columns:
        observed_budgets: set[int] = set()
        for row in records:
            try:
                observed_budgets.add(int(float(str(row.get("budget_mult", "")))))
            except ValueError:
                continue
        missing_budgets = sorted(required_budgets - observed_budgets)
        if missing_budgets:
            errors.append(f"{path.name}: missing budget_mult records {missing_budgets}")

    if (
        required_pairing_modes is not None
        and required_budgets is not None
        and {"pairing_mode", "budget_mult"} <= columns
    ):
        observed_pairs: set[tuple[str, int]] = set()
        for row in records:
            mode = str(row.get("pairing_mode", "")).strip()
            if not mode:
                continue
            try:
                budget = int(float(str(row.get("budget_mult", ""))))
            except ValueError:
                continue
            observed_pairs.add((mode, budget))
        missing_pairs = [
            f"{mode}:{budget}"
            for mode in sorted(required_pairing_modes)
            for budget in sorted(required_budgets)
            if (mode, budget) not in observed_pairs
        ]
        if missing_pairs:
            errors.append(f"{path.name}: missing pairing_mode/budget_mult records {missing_pairs}")

    if (
        required_primary_pairing_modes is not None
        and required_primary_budgets is not None
        and {"pairing_mode", "budget_mult", "is_primary"} <= columns
    ):
        primary_pairs: set[tuple[str, int]] = set()
        for row in records:
            if not _is_truthy(str(row.get("is_primary", ""))):
                continue
            mode = str(row.get("pairing_mode", "")).strip()
            if not mode:
                continue
            try:
                budget = int(float(str(row.get("budget_mult", ""))))
            except ValueError:
                continue
            primary_pairs.add((mode, budget))
        missing_primary_pairs = [
            f"{mode}:{budget}"
            for mode in sorted(required_primary_pairing_modes)
            for budget in sorted(required_primary_budgets)
            if (mode, budget) not in primary_pairs
        ]
        if missing_primary_pairs:
            errors.append(
                f"{path.name}: missing primary pairing_mode/budget_mult records "
                f"{missing_primary_pairs}"
            )
    return errors


def check_required_analysis_artifacts(
    d1_dir: Path = D1_DIR_DEFAULT,
    d2_dir: Path = D2_DIR_DEFAULT,
    d3_dir: Path = D3_DIR_DEFAULT,
    test_protocol_dir: Path = TEST_PROTOCOL_DIR_DEFAULT,
) -> list[str]:
    """Validate submission-critical D1/D2/D3 supporting-analysis artifacts."""
    csv_paths = {
        "learning_curves.csv": d2_dir / "learning_curves.csv",
        "rule_summary.csv": d2_dir / "rule_summary.csv",
        "reliability_per_method.csv": d3_dir / "reliability_per_method.csv",
        "reliability_per_method_iid.csv": d3_dir / "reliability_per_method_iid.csv",
        "reliability_ci_sensitivity.csv": d3_dir / "reliability_ci_sensitivity.csv",
        "test_set_sufficiency.csv": test_protocol_dir / "test_set_sufficiency.csv",
        "cross_metric_ranks.csv": test_protocol_dir / "cross_metric_ranks.csv",
        "cross_metric_rank_correlation.csv": test_protocol_dir / "cross_metric_rank_correlation.csv",
        "method_robustness.csv": test_protocol_dir / "method_robustness.csv",
    }
    errors: list[str] = []
    corr_schema = ANALYSIS_ARTIFACT_SCHEMAS["d1_correlation_json"]
    errors.extend(
        _check_required_json_artifact(
            d1_dir / "correlation.json",
            required=set(corr_schema.get("required", set())),
            numeric=set(corr_schema.get("numeric", set())),
            expected_values={
                key: set(value)
                for key, value in dict(corr_schema.get("expected_values", {})).items()
            },
            required_pairing_modes=set(corr_schema.get("required_pairing_modes", set())),
            required_budgets=set(corr_schema.get("required_budgets", set())),
            required_primary_pairing_modes=set(corr_schema.get("required_primary_pairing_modes", set())),
            required_primary_budgets=set(corr_schema.get("required_primary_budgets", set())),
        )
    )

    rank_schema = ANALYSIS_ARTIFACT_SCHEMAS["d1_rank_comparison.csv"]
    for mode in sorted(corr_schema.get("required_pairing_modes", set())):
        errors.extend(
            _check_required_csv_artifact(
                d1_dir / f"pairing_{mode}" / "rank_comparison.csv",
                required=set(rank_schema.get("required", set())),
                numeric=set(rank_schema.get("numeric", set())),
                expected_values={
                    key: set(value)
                    for key, value in dict(rank_schema.get("expected_values", {})).items()
                },
                required_budgets=set(rank_schema.get("required_budgets", set())),
                required_pairing_mode=str(mode),
                required_primary_budgets=set(rank_schema.get("required_primary_budgets", set())),
            )
        )

    for name, path in csv_paths.items():
        schema = ANALYSIS_ARTIFACT_SCHEMAS[name]
        errors.extend(
            _check_required_csv_artifact(
                path,
                required=set(schema.get("required", set())),
                numeric=set(schema.get("numeric", set())),
                expected_values={
                    key: set(value)
                    for key, value in dict(schema.get("expected_values", {})).items()
                },
                required_n_test=schema.get("required_n_test"),  # type: ignore[arg-type]
                required_budgets=set(schema.get("required_budgets", set())),
                check_low_n=bool(schema.get("check_low_n", False)),
            )
        )
    return errors


def check_machine_readable_root_metadata(root: Path | None = None) -> list[str]:
    """Validate root FAIR metadata before final submission sign-off."""
    root = ROOT if root is None else root
    metadata_paths = {
        ".zenodo.json": metadata_file(root, ".zenodo.json"),
        "codemeta.json": metadata_file(root, "codemeta.json"),
        "croissant-metadata.json": metadata_file(root, "croissant-metadata.json"),
        "ro-crate-metadata.json": metadata_file(root, "ro-crate-metadata.json"),
    }
    errors: list[str] = []
    for label, path in metadata_paths.items():
        if path.is_symlink():
            errors.append(f"{label} is a symlink and cannot be trusted for submission")
        elif not path.is_file():
            errors.append(f"{label} is missing from the repository root")
        elif path.stat().st_size == 0:
            errors.append(f"{label} is empty")
    if errors:
        return errors

    try:
        from scripts.check_server_return_package import check_machine_readable_metadata
    except Exception as exc:  # pragma: no cover - defensive import failure path
        return [
            "machine-readable metadata validator could not be imported: "
            f"{type(exc).__name__}: {exc}"
        ]
    return check_machine_readable_metadata(root)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog=(
            "By default, the command validates the formal full "
            "archive: benchmark/results/full_benchmark/RUN_MANIFEST.json, "
            "manuscripts/numbers.*, manuscripts/figure_manifest.json, "
            "manuscripts/figdata/, "
            "D1/D2/D3 result directories, full-sweep analysis diagnostics, "
            "and case-validation/readiness ledgers. Full manuscript sources "
            "are additionally required and are not part of the public package."
        ),
    )
    p.add_argument(
        "paper_dir",
        type=Path,
        nargs="?",
        default=PAPER_DIR_DEFAULT,
        help=f"Manuscript source directory to scan (default: {PAPER_DIR_DEFAULT}).",
    )
    p.add_argument(
        "--allow",
        type=str,
        action="append",
        default=[],
        help="Additional basenames to skip (repeatable).",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_DEFAULT,
        help="RUN_MANIFEST.json whose artifacts must be present and hash-clean.",
    )
    p.add_argument(
        "--numbers",
        type=Path,
        default=NUMBERS_DEFAULT,
        help="Manuscript numbers.json to validate against RUN_MANIFEST.",
    )
    p.add_argument(
        "--numbers-tex",
        type=Path,
        default=NUMBERS_TEX_DEFAULT,
        help="Manuscript numbers.tex whose macros must match numbers.json.",
    )
    p.add_argument(
        "--figure-manifest",
        type=Path,
        default=FIGURE_MANIFEST_DEFAULT,
        help="Manuscript figure_manifest.json to validate against local figures and figdata.",
    )
    p.add_argument(
        "--figdata-dir",
        type=Path,
        default=None,
        help="Figure-source directory whose CSV schemas must match plots_nature.py.",
    )
    p.add_argument(
        "--d1-dir",
        type=Path,
        default=D1_DIR_DEFAULT,
        help="D1 ranking-reversal directory containing correlation.json and pairing_*/rank_comparison.csv.",
    )
    p.add_argument(
        "--d2-dir",
        type=Path,
        default=D2_DIR_DEFAULT,
        help="D2 sample-complexity directory containing learning_curves.csv and rule_summary.csv.",
    )
    p.add_argument(
        "--d3-dir",
        type=Path,
        default=D3_DIR_DEFAULT,
        help="D3 reliability directory containing reliability_per_method*.csv and reliability_ci_sensitivity.csv.",
    )
    p.add_argument(
        "--test-protocol-dir",
        type=Path,
        default=TEST_PROTOCOL_DIR_DEFAULT,
        help="Full-sweep analysis directory containing test_set_sufficiency.csv.",
    )
    p.add_argument(
        "--skip-manifest-check",
        action="store_true",
        help="Only scan manuscript placeholders; do not validate RUN_MANIFEST artifacts.",
    )
    p.add_argument(
        "--skip-paper-artifact-check",
        action="store_true",
        help="Do not validate numbers.json or figure_manifest.json.",
    )
    p.add_argument(
        "--skip-source-check",
        action="store_true",
        help="Do not require materialized manuscript/cover/significance sources.",
    )
    p.add_argument(
        "--case-validation-status",
        type=Path,
        default=CASE_VALIDATION_STATUS_DEFAULT,
        help="docs/case_validation_status.csv ledger to validate before submission.",
    )
    p.add_argument(
        "--case-validation-sources",
        type=Path,
        default=CASE_VALIDATION_SOURCES_DEFAULT,
        help="docs/case_validation_sources.csv source fingerprints for the case audit ledger.",
    )
    p.add_argument(
        "--case-validation-note-sources",
        type=Path,
        default=CASE_VALIDATION_NOTE_SOURCES_DEFAULT,
        help="docs/case_validation_note_sources.csv fingerprints for human validation notes and references.",
    )
    p.add_argument(
        "--case-validation-summary",
        type=Path,
        default=CASE_VALIDATION_SUMMARY_DEFAULT,
        help="docs/CASE_VALIDATION_SUMMARY.md generated case-validation audit summary.",
    )
    p.add_argument(
        "--case-experiment-readiness",
        type=Path,
        default=CASE_EXPERIMENT_READINESS_DEFAULT,
        help="docs/case_experiment_readiness.csv generated all-case experiment-readiness ledger.",
    )
    p.add_argument(
        "--skip-case-validation-check",
        action="store_true",
        help="Do not validate case correctness or experiment-readiness audit ledgers.",
    )
    p.add_argument(
        "--skip-machine-readable-metadata-check",
        action="store_true",
        help="Do not validate root .zenodo.json, codemeta.json, Croissant, or RO-Crate metadata.",
    )
    return p.parse_args(argv)


def _filesystem_error(exc: OSError) -> str:
    target = getattr(exc, "filename", None) or getattr(exc, "filename2", None) or "<unknown>"
    return (
        f"filesystem access failed while checking {target}: "
        f"{type(exc).__name__}: {exc}. Fix the file/directory ACL or rebuild "
        "the generated artifact package, then rerun submission readiness."
    )


def _main_impl(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    skip = SKIP_BASENAMES | set(args.allow)

    paper_dir = args.paper_dir.resolve()
    if not paper_dir.is_dir():
        print(f"[error] paper directory does not exist: {paper_dir}",
              file=sys.stderr)
        return 2

    total_hits = 0
    n_scanned = 0
    source_errors: list[str] = []
    if not args.skip_source_check:
        source_errors = check_required_submission_sources(paper_dir)
        source_errors.extend(check_manuscript_case_validation_caveat(paper_dir))
        source_errors.extend(
            check_preregistration_decisions(paper_dir, args.manifest.resolve())
        )
        source_errors.extend(check_editorial_submission_materials(paper_dir))
        for err in source_errors:
            print(f"[source] {err}")

    case_errors: list[str] = []
    if not args.skip_case_validation_check:
        case_errors = check_case_validation_status(args.case_validation_status.resolve())
        case_errors.extend(
            check_case_validation_sources(
                args.case_validation_status.resolve(),
                args.case_validation_sources.resolve(),
            )
        )
        case_errors.extend(
            check_case_validation_note_sources(args.case_validation_note_sources.resolve())
        )
        case_errors.extend(
            check_case_validation_summary(
                args.case_validation_status.resolve(),
                args.case_validation_summary.resolve(),
                args.case_validation_note_sources.resolve(),
            )
        )
        case_errors.extend(
            check_case_experiment_readiness(args.case_experiment_readiness.resolve())
        )
        for err in case_errors:
            print(f"[case] {err}")

    for path in iter_candidate_files(paper_dir):
        if path.name in skip:
            continue
        n_scanned += 1
        hits = scan_file(path)
        if hits:
            rel = path.relative_to(ROOT)
            for lineno, kind, line in hits:
                print(f"{rel}:{lineno}: [{kind}] {line}")
            total_hits += len(hits)

    manifest_errors: list[str] = []
    if not args.skip_manifest_check:
        manifest_path = args.manifest.resolve()
        manifest_errors = check_manifest_artifacts(manifest_path)
        manifest_errors.extend(check_manifest_provenance(manifest_path))
        manifest_errors.extend(check_manifest_result_contract(manifest_path))
        manifest_errors.extend(check_case_source_fingerprints(manifest_path))
        manifest_errors.extend(check_method_source_fingerprints(manifest_path))
        for err in manifest_errors:
            print(f"[manifest] {err}")

    paper_artifact_errors: list[str] = []
    if not args.skip_paper_artifact_check:
        manifest_path = args.manifest.resolve()
        paper_artifact_errors.extend(
            check_numbers_consistency(args.numbers.resolve(), manifest_path)
        )
        paper_artifact_errors.extend(
            check_numbers_submission_completeness(args.numbers.resolve())
        )
        paper_artifact_errors.extend(
            check_numbers_provenance(args.numbers.resolve(), paper_dir)
        )
        paper_artifact_errors.extend(check_prereg_manifest_config(manifest_path))
        paper_artifact_errors.extend(
            check_numbers_tex_consistency(args.numbers_tex.resolve(), args.numbers.resolve())
        )
        paper_artifact_errors.extend(
            check_figure_manifest(args.figure_manifest.resolve(), paper_dir)
        )
        paper_artifact_errors.extend(check_figure_palette_contract())
        paper_artifact_errors.extend(check_required_si_files(paper_dir))
        figdata_dir = (args.figdata_dir or (paper_dir / "figdata")).resolve()
        paper_artifact_errors.extend(check_figdata_schema(figdata_dir))
        paper_artifact_errors.extend(check_figdata_manifest(figdata_dir, paper_dir))
        paper_artifact_errors.extend(check_figdata_methods(figdata_dir, manifest_path))
        paper_artifact_errors.extend(
            check_paper_artifact_lineage(
                manifest_path,
                args.numbers.resolve(),
                figdata_dir,
            )
        )
        paper_artifact_errors.extend(
            check_required_analysis_artifacts(
                args.d1_dir.resolve(),
                args.d2_dir.resolve(),
                args.d3_dir.resolve(),
                args.test_protocol_dir.resolve(),
            )
        )
        for err in paper_artifact_errors:
            print(f"[paper] {err}")

    metadata_errors: list[str] = []
    if not args.skip_machine_readable_metadata_check:
        metadata_errors = check_machine_readable_root_metadata()
        for err in metadata_errors:
            print(f"[metadata] {err}")

    print()
    print(f"[summary] scanned {n_scanned} files under {paper_dir}; "
          f"found {total_hits} unresolved manuscript token(s).")
    if not args.skip_source_check:
        print(f"[summary] source file errors: {len(source_errors)}")
    if not args.skip_case_validation_check:
        print(f"[summary] case validation errors: {len(case_errors)}")
    if not args.skip_manifest_check:
        print(f"[summary] manifest artifact errors: {len(manifest_errors)}")
    if not args.skip_paper_artifact_check:
        print(f"[summary] paper artifact errors: {len(paper_artifact_errors)}")
    if not args.skip_machine_readable_metadata_check:
        print(f"[summary] machine-readable metadata errors: {len(metadata_errors)}")
    if (
        total_hits
        or source_errors
        or case_errors
        or manifest_errors
        or paper_artifact_errors
        or metadata_errors
    ):
        print("[result] FAIL — resolve the readiness findings above before submission.")
        return 1
    print("[result] PASS — manuscript, manifest, numbers, and figure artifacts are clean.")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return _main_impl(argv)
    except PermissionError as exc:
        print(f"[filesystem] {_filesystem_error(exc)}", file=sys.stderr)
        return 1
    except OSError as exc:
        if getattr(exc, "winerror", None) == 5:
            print(f"[filesystem] {_filesystem_error(exc)}", file=sys.stderr)
            return 1
        raise


if __name__ == "__main__":
    raise SystemExit(main())
