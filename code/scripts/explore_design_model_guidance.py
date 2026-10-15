"""Exploratory, case-level design/model comparisons from the frozen sweep.

This is an analysis of registered methods, not a factorial sampler-by-model
experiment. Run from the installable code root. It reads but never alters
the frozen results and writes a compact audit to the requested output folder.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "code/benchmark/results/full_benchmark/results.csv"
CASES = ROOT / "manuscripts/figdata/fig1_case_catalog.csv"
METHODS = ROOT / "manuscripts/figdata/fig1_method_catalog.csv"
EXPECTED_RESULTS_SHA256 = "71e72d937b64abc6f5c44da2f6b2183e293e228258ce4c0abce9bb4100f0ca0b"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--table-output", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    sha = hashlib.sha256(RESULTS.read_bytes()).hexdigest()
    if sha != EXPECTED_RESULTS_SHA256:
        raise ValueError(f"Frozen results fingerprint changed: {sha}")
    case_sha = hashlib.sha256(CASES.read_bytes()).hexdigest()
    method_sha = hashlib.sha256(METHODS.read_bytes()).hexdigest()

    case_meta = pd.read_csv(CASES).set_index("case_id")
    method_meta = pd.read_csv(METHODS).set_index("method")
    usecols = ["case_id", "method", "budget_mult", "repeat", "status", "nrmse"]
    data = pd.read_csv(RESULTS, usecols=usecols)
    data = data[data.budget_mult.isin([10, 20])].copy()
    data["nrmse"] = pd.to_numeric(data.nrmse, errors="coerce")
    data["valid"] = data.status.eq("ok") & data.nrmse.notna()
    medians = (
        data[data.valid]
        .groupby(["case_id", "budget_mult", "method"], observed=True)
        .nrmse.median()
        .rename("median_nrmse")
    )
    counts = (
        data.groupby(["case_id", "budget_mult", "method"], observed=True)
        .valid.agg(["sum", "count"])
        .rename(columns={"sum": "n_ok", "count": "n_attempted"})
    )
    panel = counts.join(medians).reset_index()
    panel["rank"] = panel.groupby(["case_id", "budget_mult"]).median_nrmse.rank(
        method="min", na_option="bottom"
    )
    panel = panel.join(case_meta[["domain", "dim", "topology"]], on="case_id")
    panel = panel.join(method_meta[["family", "surrogate"]], on="method")
    panel["strong_interaction"] = panel.topology.str.contains("strong_interaction", na=False)
    panel["sparse_effects"] = panel.topology.str.contains("sparse_effects", na=False)
    panel["higher_dim"] = panel.dim.ge(10)
    panel.to_csv(output / "case_method_medians.csv", index=False)

    groups = {
        "all": panel.case_id.notna(),
        "strong_interaction": panel.strong_interaction,
        "without_strong_interaction": ~panel.strong_interaction,
        "sparse_effects": panel.sparse_effects,
        "without_sparse_effects": ~panel.sparse_effects,
        "dimension_10_plus": panel.higher_dim,
    }
    summaries = []
    for label, mask in groups.items():
        section = panel[mask]
        for (budget, method), subset in section.groupby(["budget_mult", "method"]):
            summaries.append(
                {
                    "group": label,
                    "budget_mult": int(budget),
                    "method": method,
                    "n_cases": subset.case_id.nunique(),
                    "mean_case_rank": subset["rank"].mean(),
                    "median_case_rank": subset["rank"].median(),
                    "median_case_nrmse": subset.median_nrmse.median(),
                    "n_case_wins": int(subset["rank"].eq(1).sum()),
                    "n_ok": int(subset.n_ok.sum()),
                    "n_attempted": int(subset.n_attempted.sum()),
                }
            )
    summary = pd.DataFrame(summaries).sort_values(
        ["group", "budget_mult", "mean_case_rank", "method"]
    )
    summary.to_csv(output / "group_method_summary.csv", index=False)

    if args.table_output:
        names = {
            "lhs_opt+gp": "Optimised LHS + GP",
            "max_variance": "Maximum variance + GP",
            "sobol+gp": "Sobol' + GP",
            "halton+gp": "Halton + GP",
        }
        labels = {
            "all": "All registered cases",
            "strong_interaction": "Strong-interaction label",
            "sparse_effects": "Sparse-effects label",
        }
        lines = [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{\textbf{Supplementary Table S14.} Exploratory design--surrogate guidance from the frozen physical-case sweep. Each entry gives the first and second method by mean within-case rank (lower is better) of median NRMSE over successful repeats. Topology labels overlap; the strong-interaction label denotes case metadata rather than the separate Sobol' threshold. This is a descriptive ranking of fixed bundles, not an estimate of independent design or model effects.}",
            r"\label{tab:si_design_model_guidance}",
            r"\small",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{tabular}{@{}lrrll@{}}",
            r"\toprule",
            r"Group & Cases & Budget & First (rank) & Second (rank) \\",
            r"\midrule",
        ]
        for group in labels:
            for budget in (10, 20):
                top = summary[
                    summary.group.eq(group) & summary.budget_mult.eq(budget)
                ].head(2)
                if len(top) != 2:
                    raise ValueError(f"Incomplete summary for {group} at {budget}d")
                first, second = top.iloc[0], top.iloc[1]
                lines.append(
                    f"{labels[group]} & {int(first.n_cases)} & ${budget}d$ & "
                    f"{names[first.method]} ({first.mean_case_rank:.2f}) & "
                    f"{names[second.method]} ({second.mean_case_rank:.2f}) " + r"\\"
                )
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        dest = args.table_output.resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    comparisons = [
        ("lhs+gp", "sobol+gp"),
        ("lhs_opt+gp", "sobol+gp"),
        ("lhs_opt+gp", "lhs_opt+gp_rbf"),
        ("lhs_opt+gp", "lhs_opt+rf"),
        ("lhs_opt+gp", "lhs_opt+poly3"),
        ("lhs_opt+gp", "lhs_opt+poly5"),
        ("lhs_opt+gp", "max_variance"),
        ("lhs_opt+gp", "botorch_saasbo"),
    ]
    pair_rows = []
    for label, mask in groups.items():
        subset = panel[mask]
        for budget in [10, 20]:
            wide = subset[subset.budget_mult.eq(budget)].pivot(
                index="case_id", columns="method", values="median_nrmse"
            )
            for left, right in comparisons:
                paired = wide[[left, right]].dropna()
                pair_rows.append(
                    {
                        "group": label,
                        "budget_mult": budget,
                        "left": left,
                        "right": right,
                        "n_paired_cases": len(paired),
                        "left_better": int((paired[left] < paired[right]).sum()),
                        "right_better": int((paired[right] < paired[left]).sum()),
                        "ties": int((paired[right] == paired[left]).sum()),
                        "median_ratio_left_over_right": (
                            (paired[left] / paired[right]).median() if len(paired) else float("nan")
                        ),
                    }
                )
    pd.DataFrame(pair_rows).to_csv(output / "paired_comparisons.csv", index=False)
    (output / "README.md").write_text(
        "# Exploratory design–surrogate guidance audit\n\n"
        f"Frozen result SHA-256: `{sha}`. No raw results were changed.\n\n"
        f"Case catalog SHA-256: `{case_sha}`. Method catalog SHA-256: `{method_sha}`.\n\n"
        "Case-level median NRMSE is computed over successful repeats at 10d and 20d. "
        "Each case ranks the 20 registered methods by that median; methods with no "
        "successful repeat rank last. Group summaries average ranks across cases, "
        "and paired comparisons use only cases where both medians exist. "
        "Topology labels overlap and are case-level descriptors. A method name "
        "identifies a fixed design–surrogate bundle; the pool is not a full "
        "factorial crossing, so design/model interactions are not identifiable. "
        "These analyses are post hoc and descriptive.\n\n"
        "Reproduce from `code/` in a disposable copy of the workspace:\n\n"
        "```powershell\n"
        "python scripts/explore_design_model_guidance.py --output-dir ../project_docs/manuscript_revision_20260926/design_model_guidance --table-output ../manuscripts/tables/si_design_model_guidance.tex\n"
        "```\n\n"
        "The output contains all 34 case-by-method medians, group summaries "
        "and paired comparisons; Supplementary Table S14 is generated from "
        "the same group summary.\n",
        encoding="utf-8",
    )
    print("cases:", case_meta.shape[0])
    for label, mask in groups.items():
        print(label, panel.loc[mask, "case_id"].nunique())
        print(summary[(summary.group.eq(label)) & (summary.budget_mult.eq(10))].head(6)[
            ["method", "mean_case_rank", "n_case_wins", "n_ok", "n_attempted"]
        ].to_string(index=False))
    print("result sha256:", sha)


if __name__ == "__main__":
    main()
