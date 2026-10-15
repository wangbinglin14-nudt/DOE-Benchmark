"""
generate_si.py
==============

Auto-generate the Supplementary Information (SI) tables for the
Communications Engineering manuscript directly from the canonical sources
of truth in the codebase:

* **SI Table S1 — Method hyperparameters** ← ``benchmark.hyperparameters.HYPERPARAMETERS``
* **SI Table S2 — Case Sobol summary**     ← current-source corrected artifact
* **SI Table S3 — Case characterisation**  ← ``CaseMeta.{domain, dim_input,
    governing_equation, response_topology, has_failure_boundary, ...}``
* **SI Note S4 — Multiplicity policy**     ← preregistered Holm family and
    sensitivity rationale
* **SI Note S5 — D2 policy**               ← exploratory topology and
    penalised-denominator reporting contract

Each table is emitted in two formats:

* ``paper/tables/si_<id>.tex`` — booktabs-style LaTeX for direct inclusion
* ``paper/tables/si_<id>.md`` — companion Markdown

By regenerating SI from code we guarantee that the manuscript can
never silently drift from the implementation — the same property
enforced by the CI hyperparameter-locking test.

Usage
-----

    python scripts/generate_si.py                 # write to paper/tables/si_*.tex|md
    python scripts/generate_si.py --output-dir docs/SI

"""

from __future__ import annotations

import argparse
import csv
import io
import shutil
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Force UTF-8 stdout on Windows so the unicode glyphs render correctly.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def _esc_tex(s: str) -> str:
    """Minimal LaTeX escaping for booktabs cells."""
    if s is None:
        return "--"
    return (
        str(s)
        .replace("\\", "\\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def _kwargs_to_str(d: Dict[str, Any], joiner: str = ", ") -> str:
    if not d:
        return "--"
    parts = []
    for k, v in d.items():
        if isinstance(v, tuple):
            v = "(" + ",".join(map(str, v)) + ")"
        elif isinstance(v, float) and abs(v) < 1e-2:
            v = f"{v:.0e}"
        parts.append(f"{k}={v}")
    return joiner.join(parts)


def _case_validation_evidence_tiers() -> Dict[str, str]:
    """Return case_id -> evidence tier from the case-validation ledger."""
    status_csv = ROOT / "docs" / "case_validation_status.csv"
    if not status_csv.exists():
        warnings.warn("case-validation status CSV missing; SI S3 evidence tiers set to not_listed")
        return {}
    try:
        from scripts.generate_case_validation_summary import evidence_tier

        with status_csv.open("r", encoding="utf-8", newline="") as f:
            return {
                row["case_id"]: evidence_tier(row)
                for row in csv.DictReader(f)
                if row.get("case_id")
            }
    except Exception as exc:
        warnings.warn(
            "could not compute case-validation evidence tiers for SI S3: "
            f"{type(exc).__name__}: {exc}"
        )
        return {}


def _load_corrected_sobol() -> Dict[str, Dict[str, Any]]:
    """Load the uniform current-source Sobol artifact used by SI Table S2.

    Refusing to fall back to ``CaseMeta`` is deliberate: those historical
    metadata values predate several case-source changes and are not a valid
    quantitative source for the submission tables.
    """
    path = ROOT / "benchmark" / "data" / "sobol_corrected_full_current_source.json"
    if not path.exists():
        raise RuntimeError(
            "corrected Sobol artifact missing; run scripts/recompute_sobol_all_current.py"
        )
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("N") != 1024 or payload.get("seed") != 20260831:
        raise RuntimeError("corrected Sobol artifact has unexpected N/seed")
    rows = payload.get("cases", [])
    if len(rows) != 34 or len({r.get("case_id") for r in rows}) != 34:
        raise RuntimeError("corrected Sobol artifact must contain 34 unique cases")
    if any(not r.get("S1_conf") or not r.get("ST_conf") for r in rows):
        raise RuntimeError("corrected Sobol artifact is missing confidence intervals")
    return {r["case_id"]: r for r in rows}
# Stems that build_supplementary.py consumes to produce the *_clean.tex files
# that supplementary.tex actually inputs.  They must not be visited by
# \input directly, because their table numbers duplicate the clean versions.
_INTERMEDIATE_STEMS = {"si_cases", "si_sobol", "si_hyperparameters"}
_INTERMEDIATE_NOTICE = (
    "% GENERATED by scripts/generate_si.py -- intermediate table source.\n"
    "% Do NOT \\input this file: supplementary.tex inputs the *_clean.tex\n"
    "% counterpart written by scripts/build_supplementary.py.\n"
)


def _save_pair(out_dir: Path, stem: str, latex: str, markdown: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if stem in _INTERMEDIATE_STEMS:
        latex = _INTERMEDIATE_NOTICE + latex
    (out_dir / f"{stem}.tex").write_text(latex, encoding="utf-8")
    (out_dir / f"{stem}.md").write_text(markdown, encoding="utf-8")
    print(f"  [ok] {stem}.tex + {stem}.md")


SI_FILES = (
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


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _replace_files_atomically(src_dir: Path, dst_dir: Path, filenames: Tuple[str, ...]) -> None:
    """Replace the generated SI files as one submission package."""
    src_dir = src_dir.resolve()
    dst_dir = dst_dir.resolve()
    dst_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = Path(tempfile.mkdtemp(prefix=".si.backup.", dir=str(dst_dir)))
    moved: list[str] = []
    try:
        for name in filenames:
            src = src_dir / name
            dst = dst_dir / name
            if not src.exists() or src.stat().st_size == 0:
                raise RuntimeError(f"generated SI artifact missing or empty: {src}")
            if dst.exists():
                dst.rename(backup_dir / name)
            shutil.move(str(src), str(dst))
            moved.append(name)
    except Exception:
        for name in moved:
            _remove_path(dst_dir / name)
        for backup in backup_dir.iterdir():
            backup.rename(dst_dir / backup.name)
        raise
    finally:
        if backup_dir.exists():
            shutil.rmtree(backup_dir)


def _build_si(out_dir: Path) -> None:
    s1 = _make_table_s1()
    if s1:
        _save_pair(out_dir, "si_hyperparameters", *s1)

    s2 = _make_table_s2()
    if s2:
        _save_pair(out_dir, "si_sobol", *s2)

    s3 = _make_table_s3()
    if s3:
        _save_pair(out_dir, "si_cases", *s3)

    _save_pair(out_dir, "si_multiple_testing", *_make_multiple_testing_note())
    _save_pair(out_dir, "si_d2_policy", *_make_d2_policy_note())


def write_si_package(out_dir: Path) -> None:
    """Generate the canonical SI files and replace them only after success."""
    out_dir = out_dir.resolve()
    if (out_dir / "tables").is_dir():
        out_dir = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_parent = Path(tempfile.mkdtemp(
        prefix=".si.tmp.",
        dir=str(out_dir),
    ))
    try:
        _build_si(tmp_parent)
        _replace_files_atomically(tmp_parent, out_dir, SI_FILES)
    finally:
        if tmp_parent.exists():
            shutil.rmtree(tmp_parent)


# ─────────────────────────────────────────────────────────────────────
# SI Table S1 — Method hyperparameters
# ─────────────────────────────────────────────────────────────────────
def _make_table_s1() -> Tuple[str, str]:
    from benchmark.hyperparameters import HYPERPARAMETERS

    rows: List[Tuple[str, str, str, str]] = []
    for name, entry in sorted(HYPERPARAMETERS.items()):
        sampler = entry.get("sampler", "--")
        surrogate = entry.get("surrogate", "--")
        kwargs = entry.get("surrogate_kwargs") or entry.get("method_kwargs") or {}
        rows.append((name, sampler, surrogate, _kwargs_to_str(kwargs)))

    # Markdown
    md_lines = [
        "# Supplementary Table S1 · Method hyperparameters",
        "",
        "_All methods are evaluated with the same hyperparameter set across_",
        "_every case; this table is auto-generated from_",
        "`benchmark/hyperparameters.py` _and locked by the CI test_",
        "`benchmark/tests/test_hyperparameters.py`.",
        "",
        "| Method | Sampler | Surrogate / kernel | Hyperparameters |",
        "|---|---|---|---|",
    ]
    for n, sam, sur, kw in rows:
        md_lines.append(f"| `{n}` | {sam} | {sur} | {kw} |")
    markdown = "\n".join(md_lines) + "\n"

    # LaTeX
    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{\textbf{Supplementary Table S1.} Method hyperparameters.  Auto-generated from \texttt{benchmark/hyperparameters.py} and locked against drift by the CI test \texttt{benchmark/\allowbreak tests/\allowbreak test\_hyperparameters.py}.}",
        r"\label{tab:si_hyperparameters}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}llll@{}}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Sampler} & \textbf{Surrogate / kernel} & \textbf{Hyperparameters} \\",
        r"\midrule",
    ]
    for n, sam, sur, kw in rows:
        tex_lines.append(
            f"\\texttt{{{_esc_tex(n)}}} & {_esc_tex(sam)} & "
            f"{_esc_tex(sur)} & {_esc_tex(kw)} \\\\"
        )
    tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    latex = "\n".join(tex_lines) + "\n"

    return latex, markdown


# ─────────────────────────────────────────────────────────────────────
# SI Table S2 — Sobol summary per case
# ─────────────────────────────────────────────────────────────────────
def _make_table_s2() -> Optional[Tuple[str, str]]:
    try:
        from doe_testbed import list_cases
    except ImportError:
        warnings.warn("doe_testbed not importable; skipping SI Table S2")
        return None

    corrected = _load_corrected_sobol()
    rows = []
    for meta in list_cases():
        if isinstance(meta, str):
            from doe_testbed import get_case
            meta = get_case(meta).meta
        try:
            sobol = corrected[meta.case_id]
        except KeyError as exc:
            raise RuntimeError(f"corrected Sobol artifact missing {meta.case_id}") from exc
        st = sobol["ST"]
        s1 = sobol["S1"]
        if st:
            top_var, top_st = max(st.items(), key=lambda kv: kv[1])
            sum_st = sum(st.values())
            second_order = max(0.0, sum_st - sum(s1.values()))
        else:
            top_var, top_st, sum_st, second_order = "--", float("nan"), float("nan"), float("nan")
        rows.append({
            "case_id": meta.case_id,
            "domain": getattr(meta, "domain", "--"),
            "d": int(getattr(meta, "dim_input", 0)),
            "topology": ", ".join(getattr(meta, "response_topology", []) or ["--"]),
            "top_var": top_var,
            "top_st": top_st,
            "sum_st": sum_st,
            "interactions": second_order,
        })
    rows.sort(key=lambda r: (r["domain"], r["d"], r["case_id"]))

    # Markdown
    md_lines = [
        "# Supplementary Table S2 · Per-case Sobol' characterisation",
        "",
        "_Auto-generated from `benchmark/data/sobol_corrected_full_current_source.json` —_",
        "_SALib Saltelli sampling, N=1024, seed=20260831; see Methods §M1._",
        "",
        "| Case | Domain | d | Topology | Top-effect var | $S_T$(top) | $\\Sigma S_T$ | Interaction mass |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md_lines.append(
            f"| `{r['case_id']}` | {r['domain']} | {r['d']} | {r['topology']} | "
            f"`{r['top_var']}` | {r['top_st']:.3f} | {r['sum_st']:.3f} | "
            f"{r['interactions']:.3f} |"
            if r["top_st"] == r["top_st"] else  # NaN check
            f"| `{r['case_id']}` | {r['domain']} | {r['d']} | {r['topology']} | -- | -- | -- | -- |"
        )
    markdown = "\n".join(md_lines) + "\n"

    # LaTeX
    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{\textbf{Supplementary Table S2.} Per-case Sobol' characterisation.}",
        r"\label{tab:si_sobol}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lllclrrr@{}}",
        r"\toprule",
        r"\textbf{Case} & \textbf{Domain} & $d$ & \textbf{Topology} &"
        r" \textbf{Top var} & $S_T$ & $\sum S_T$ & \textbf{Inter.} \\",
        r"\midrule",
    ]
    for r in rows:
        if r["top_st"] != r["top_st"]:
            tex_lines.append(
                f"\\texttt{{{_esc_tex(r['case_id'])}}} & {_esc_tex(r['domain'])} & "
                f"{r['d']} & {_esc_tex(r['topology'])} & -- & -- & -- & -- \\\\"
            )
        else:
            tex_lines.append(
                f"\\texttt{{{_esc_tex(r['case_id'])}}} & {_esc_tex(r['domain'])} & "
                f"{r['d']} & {_esc_tex(r['topology'])} & "
                f"\\texttt{{{_esc_tex(r['top_var'])}}} & "
                f"{r['top_st']:.3f} & {r['sum_st']:.3f} & "
                f"{r['interactions']:.3f} \\\\"
            )
    tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    latex = "\n".join(tex_lines) + "\n"

    return latex, markdown


# ─────────────────────────────────────────────────────────────────────
# SI Table S3 — Case characterisation (governing equation summary)
# ─────────────────────────────────────────────────────────────────────
def _make_table_s3() -> Optional[Tuple[str, str]]:
    try:
        from doe_testbed import list_cases
    except ImportError:
        warnings.warn("doe_testbed not importable; skipping SI Table S3")
        return None
    evidence_by_case = _case_validation_evidence_tiers()

    rows = []
    for meta in list_cases():
        if isinstance(meta, str):
            from doe_testbed import get_case
            meta = get_case(meta).meta
        rows.append({
            "case_id": meta.case_id,
            "domain": getattr(meta, "domain", "--"),
            "d": int(getattr(meta, "dim_input", 0)),
            "m": int(getattr(meta, "dim_output", 1)),
            "pde": getattr(meta, "pde_type", "--"),
            "governing": (getattr(meta, "governing_equation", "") or "").split("\n")[0][:60],
            "cost": getattr(meta, "cost_level", "--"),
            "failure": "✓" if getattr(meta, "has_failure_boundary", False) else "—",
            "evidence": evidence_by_case.get(meta.case_id, "not_listed"),
        })
    rows.sort(key=lambda r: (r["domain"], r["d"], r["case_id"]))

    # Markdown
    md_lines = [
        "# Supplementary Table S3 · Case characterisation",
        "",
        "_Auto-generated from `CaseMeta` and `docs/case_validation_status.csv` —_",
        "_see `doe_testbed/cases/` for full implementations and_",
        "_`docs/CASE_VALIDATION_SUMMARY.md` for evidence-tier definitions._",
        "",
        "> Note: `d` is the number of input dimensions. A case-ID suffix such as `_2d` may denote the physical spatial dimension and is therefore not necessarily equal to `d`.",
        "",
        "| Case | Domain | d | m | PDE | Governing eq. | Cost | Failure? | Evidence tier |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md_lines.append(
            f"| `{r['case_id']}` | {r['domain']} | {r['d']} | {r['m']} | "
            f"{r['pde']} | {r['governing']} | {r['cost']} | {r['failure']} |"
            f" `{r['evidence']}` |"
        )
    markdown = "\n".join(md_lines) + "\n"

    # LaTeX
    tex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{\textbf{Supplementary Table S3.} Case characterisation.}",
        r"\label{tab:si_cases}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}llcclllcl@{}}",
        r"\toprule",
        r"\textbf{Case} & \textbf{Domain} & $d$ & $m$ & \textbf{PDE} &"
        r" \textbf{Governing eq.} & \textbf{Cost} & \textbf{Failure} & \textbf{Evidence tier} \\",
        r"\midrule",
    ]
    for r in rows:
        failure_tex = r["failure"].replace("✓", r"$\checkmark$").replace("—", "--")
        tex_lines.append(
            f"\\texttt{{{_esc_tex(r['case_id'])}}} & {_esc_tex(r['domain'])} & "
            f"{r['d']} & {r['m']} & {_esc_tex(r['pde'])} & "
            f"{_esc_tex(r['governing'])} & {_esc_tex(r['cost'])} & "
            f"{failure_tex} & "
            f"\\texttt{{{_esc_tex(r['evidence'])}}} \\\\"
        )
    tex_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        r"\smallskip",
        r"\noindent\textit{Note.} The column $d$ is the number of input dimensions. A case-ID suffix such as \texttt{\_2d} may denote the physical spatial dimension and is therefore not necessarily equal to $d$.",
    ]
    latex = "\n".join(tex_lines) + "\n"

    return latex, markdown


# ─────────────────────────────────────────────────────────────────────
# SI Note S4 — Multiplicity policy
# ─────────────────────────────────────────────────────────────────────
def _make_multiple_testing_note() -> Tuple[str, str]:
    """Document the preregistered multiplicity policy for submission."""
    markdown = "\n".join([
        "# Supplementary Table S4 · Multiple-testing policy",
        "",
        "The confirmatory analysis uses Holm-Bonferroni control of the",
        "family-wise error rate at alpha = 0.05 across the three registered",
        "discovery families D1, D2 and D3. The ordered thresholds are",
        "0.05/3, 0.05/2 and 0.05.",
        "",
        "D1 reports Spearman rho as the effect-size estimator with bootstrap",
        "confidence intervals. D2 uses a one-sided Clopper-Pearson upper",
        "confidence bound for the pre-specified topology pass-rate criterion,",
        "so no topology-level Holm correction is applied. D3 uses method-level",
        "Holm adjustment for signed",
        "bias tests and reports cluster-aware confidence intervals as the",
        "primary uncertainty estimate.",
        "",
        "Bonferroni and BH-FDR are reserved for sensitivity reporting: they may",
        "be tabulated after the full sweep, but they do not replace the",
        "pre-registered Holm decision rule or headline claims.",
        "",
    ]) + "\n"

    latex = "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{\textbf{Supplementary Table S4.} Multiple-testing policy.}",
        r"\label{tab:si_multiple_testing}",
        r"\footnotesize",
        r"\begin{tabular}{@{}lp{0.68\linewidth}@{}}",
        r"\toprule",
        r"\textbf{Item} & \textbf{Policy} \\",
        r"\midrule",
        (
            r"Family-wise rule & Holm--Bonferroni at $\alpha=0.05$ across "
            r"D1, D2 and D3; ordered thresholds are $0.05/3$, $0.05/2$ and "
            r"$0.05$. \\"
        ),
        (
            r"D1 & Spearman $\rho$ is the effect-size estimator; bootstrap "
            r"confidence intervals quantify precision. \\"
        ),
        (
            r"D2 & The primary criterion is a one-sided Clopper--Pearson upper "
            r"confidence bound on the topology pass rate, so no topology-level Holm correction is "
            r"applied. \\"
        ),
        (
            r"D3 & Signed-bias tests use method-level Holm adjustment; "
            r"cluster-aware confidence intervals are primary. \\"
        ),
        (
            r"Sensitivity & Bonferroni and BH-FDR are reserved for sensitivity "
            r"tables after the full sweep and do not replace the "
            r"pre-registered Holm decision rule. \\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]) + "\n"

    return latex, markdown


def _make_d2_policy_note() -> Tuple[str, str]:
    """Document the D2 exploratory-topology and denominator policy."""
    markdown = "\n".join([
        "# Supplementary Table S5 · D2 topology and denominator policy",
        "",
        "The D2 headline analysis uses `penalized_failed_in_denominator`.",
        "All 20 methods remain in each topology's denominator. Error medians",
        "pool successful runs; a method without a qualifying median at any",
        "tested budget is non-passing. Failed and timed-out runs are counted",
        "separately, without a synthetic NRMSE penalty.",
        "",
        "The D2 accuracy target is written \u03b5 (epsilon), with \u03b5 = 0.20 for the",
        "registered analysis. The symbol \u03c4 (tau) is reserved for the D1 Kendall",
        "rank correlation; the two are deliberately distinct.",
        "",
        "Topologies with fewer than three registered cases are flagged",
        "`exploratory=True` in `rule_summary.csv`. These rows are reported",
        "separately and are not counted in the headline numerator or",
        "denominator for the 10d rule.",
        "",
        "The table S5 contract is the `rule_summary.csv` schema:",
        "`topology`, `methods_total`, `methods_pass_at_10d`,",
        "`frac_pass_at_10d`, `median_n_star`, `max_n_star`,",
        "`n_methods_never_pass`, `n_cases`, `case_ids`, `exploratory`,",
        "`n_failed_cells_per_topology`, `analysis_type` and `is_primary`.",
        "",
    ]) + "\n"

    latex = "\n".join([
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{\textbf{Supplementary Table S5.} D2 topology and denominator policy.}",
        r"\label{tab:si_d2_policy}",
        r"\footnotesize",
        r"\begin{tabular}{@{}lp{0.68\linewidth}@{}}",
        r"\toprule",
        r"\textbf{Item} & \textbf{Policy} \\",
        r"\midrule",
        (
            r"Primary analysis & The D2 headline uses "
            r"\texttt{penalized\_failed\_in\_denominator}. All 20 methods remain "
            r"in each topology's denominator. Error medians pool successful runs; "
            r"a method without a qualifying median at any tested budget is non-passing. "
            r"Failed and timed-out runs are counted separately, without a synthetic NRMSE penalty. \\"
        ),
        (
            r"Exploratory bucket & Topologies with fewer than three registered "
            r"cases are flagged \texttt{exploratory=True}; they are reported "
            r"separately and not counted in the headline numerator or "
            r"denominator. \\"
        ),
        (
            r"Accuracy target & Written $\varepsilon$, with $\varepsilon=0.20$ "
            r"for the registered analysis. $\tau$ denotes the D1 Kendall rank "
            r"correlation and is never used for the D2 target. \\"
        ),
        (
            r"Table S5 contract & The table is generated from "
            r"\texttt{rule\_summary.csv} with topology, pass-rate, "
            r"sample-complexity, \texttt{n\_cases}, \texttt{case\_ids}, "
            r"\texttt{exploratory}, failed-cell, \texttt{analysis\_type} and "
            r"\texttt{is\_primary} fields. \\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]) + "\n"

    return latex, markdown


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Auto-generate Supplementary Information tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "paper",
                   help="Where to write si_*.tex and si_*.md.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.output_dir)
    print(f"Generating Supplementary Information into {out_dir}")
    print()
    try:
        write_si_package(out_dir)
    except Exception as exc:
        print(f"[ERROR] SI package generation failed; existing SI files unchanged: {exc}")
        return 1

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
