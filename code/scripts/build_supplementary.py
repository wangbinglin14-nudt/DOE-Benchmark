# -*- coding: utf-8 -*-
"""Regenerate the Supplementary Information tables from archived data artifacts.

Reads only committed analysis outputs (no hand-entered numbers) and writes the
LaTeX table fragments consumed by paper/supplementary.tex:

  S1  si_hyperparameters_clean.tex   (wrapping method/hyperparameter table)
  S2  si_sobol_clean.tex             (wrapping per-case Sobol' table)
  S3  si_cases_clean.tex             (case-characterisation ledger)
  S6  si_d3_per_method.tex           (per-method P_f calibration)
  S7  si_d3_ci_sensitivity.tex       (cluster vs i.i.d. bootstrap CI widths)
  S8  si_d3_case_level.tex            (case-level D3 sensitivity)
  S9  si_mc_noise.tex                 (test-set Monte-Carlo noise floor)

S4 and S5 are static policy notes (si_multiple_testing.tex, si_d2_policy.tex).

Usage:  python scripts/build_supplementary.py
"""
import csv, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
PAPER = os.path.join(ROOT, "paper")
TABLES = os.path.join(PAPER, "tables") if os.path.isdir(os.path.join(PAPER, "tables")) else PAPER
REL = os.path.join(ROOT, "benchmark", "results", "reliability_recovery")
FULL = os.path.join(ROOT, "benchmark", "results", "full_benchmark", "analysis")

from paper_names import (
    case_label, domain_label, method_label, sampler_label,
    surrogate_label, tier_label, topology_label,
)


def esc(s):
    return s.replace("_", r"\_")


def unesc(s):
    """Strip \\texttt{...} wrappers and LaTeX escapes from a scraped cell."""
    return (
        s.replace(r"\texttt{", "").replace("}", "").replace(r"\_", "_").strip()
    )


def rd(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def write(name, lines):
    p = os.path.join(TABLES, name)
    open(p, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("wrote", name, len(lines), "lines")


# ---------- S6: per-method D3 signed bias ----------
fam_abbr = {
    "sequential_botorch_saas": "BoTorch SAAS",
    "static_design": "Static design",
    "sequential_ego": "EGO",
    "sequential_botorch_ei": "BoTorch EI",
    "sequential_active_learning": "Active learning",
    "sequential_reliability": "Reliability (AK-MCS)",
}
rows = rd(os.path.join(REL, "reliability_per_method.csv"))
rows.sort(key=lambda r: float(r["signed_bias_mean"]))
out = [r"\begin{table}[htbp]", r"\centering",
       r"\caption{\textbf{Supplementary Table S6.} Per-method probability-of-failure calibration at the $10d$ budget. Signed bias $\Delta=\widehat{P}_f-P_f$ (positive $=$ over-estimation for the selected output) with its case-cluster bootstrap 95\% CI and one-sided $p$-value adjusted by Holm within the 20-method D3 family, plus held-out balanced accuracy. All \NumDThreeNTested{} eligible methods are significant after this method-level correction for the registered per-cell estimand; these $p$-values are not independent-case evidence. Generated from \texttt{reliability\_per\_method.csv}.}",
       r"\label{tab:si_d3_per_method}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
       r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
       r"\textbf{Method} & \textbf{Family} & $n$ & \textbf{Signed bias [95\% CI]} & \textbf{$p$ (Holm)} & \textbf{Bal.\ acc.} \\",
       r"\midrule"]
for r in rows:
    fam = fam_abbr.get(r["method_family"], esc(r["method_family"]))
    sb = "%.4f [%.4f, %.4f]" % (float(r["signed_bias_mean"]), float(r["signed_bias_ci_lo"]), float(r["signed_bias_ci_hi"]))
    out.append(r"%s & %s & %s & %s & %.1e & %.3f \\" % (
        esc(method_label(r["method"])), fam, r["n_runs"], sb, float(r["p_holm"]), float(r["f_bal_acc_mean"])))
out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write("si_d3_per_method.tex", out)

# ---------- S7: cluster vs iid CI sensitivity ----------
rows = rd(os.path.join(REL, "reliability_ci_sensitivity.csv"))
rows.sort(key=lambda r: float(r["signed_bias_ci_width_ratio_cluster_over_iid"]), reverse=True)
out = [r"\begin{table}[htbp]", r"\centering",
       r"\caption{\textbf{Supplementary Table S7.} Cluster-vs-i.i.d.\ bootstrap sensitivity for the D3 signed bias. The case-cluster bootstrap (primary estimator) is compared against an i.i.d.\ bootstrap; the ratio of 95\% CI widths quantifies the variance inflation from within-case correlation. The cluster CI is uniformly wider, so the i.i.d.\ interval would overstate precision. Generated from \texttt{reliability\_ci\_sensitivity.csv}.}",
       r"\label{tab:si_d3_ci_sensitivity}", r"\footnotesize", r"\setlength{\tabcolsep}{5pt}",
       r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
       r"\textbf{Method} & \textbf{CI width (cluster)} & \textbf{CI width (i.i.d.)} & \textbf{Ratio} \\",
       r"\midrule"]
for r in rows:
    out.append(r"%s & %.4f & %.4f & %.2f$\times$ \\" % (
        esc(method_label(r["method"])), float(r["signed_bias_ci_width_cluster"]),
        float(r["signed_bias_ci_width_iid"]), float(r["signed_bias_ci_width_ratio_cluster_over_iid"])))
out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write("si_d3_ci_sensitivity.tex", out)

# ---------- S8: case-level D3 sensitivity ----------
case_level = json.load(open(os.path.join(REL, "d3_case_level_sensitivity.json"), encoding="utf-8"))
case_rows = rd(os.path.join(REL, "d3_case_level_sensitivity.csv"))
_n_cases = int(case_level["n_cases"])
_n_positive = int(case_level["n_positive_cases"])
_s8_caption = (
    r"\caption{\textbf{Supplementary Table S8.} Post-hoc case-level sensitivity "
    r"for D3 at the $10d$ budget. Signed bias is first averaged within each "
    r"physical case, so the %d case means are the independent units for the "
    r"one-sided Wilcoxon test. %d of %d case means are positive; the equal-case "
    r"bootstrap interval is reported descriptively. This sensitivity analysis "
    r"is not added to the preregistered Holm discovery family. Generated from "
    r"\texttt{d3\_case\_level\_sensitivity.csv}.}" % (_n_cases, _n_positive, _n_cases)
)
out = [r"\begin{table}[htbp]", r"\centering",
       _s8_caption,
       r"\label{tab:si_d3_case_level}", r"\footnotesize", r"\setlength{\tabcolsep}{5pt}",
       r"\begin{tabular}{@{}lrr@{}}", r"\toprule",
       r"\textbf{Case} & \textbf{$n$ runs} & \textbf{Mean signed bias} \\", r"\midrule"]
for r in case_rows:
    out.append(r"%s & %s & %.4f \\" % (esc(case_label(r["case_id"])), r["n_runs"], float(r["mean_signed_bias"])))
out += [r"\midrule",
        r"Equal-case mean (all %d cases) & -- & %.4f \\" % (case_level["n_cases"], case_level["equal_case_mean_signed_bias"]),
        r"Equal-case bootstrap 95\%% CI & -- & [%.4f, %.4f] \\" % (case_level["equal_case_bootstrap_ci_lo"], case_level["equal_case_bootstrap_ci_hi"]),
        r"Case-level Wilcoxon $p$ (one-sided) & -- & %.4f \\" % case_level["wilcoxon_case_level_one_sided_greater_p"],
        r"Holm threshold (family-wise) & -- & %.4f \\" % case_level["holm_family_threshold"],
        r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write("si_d3_case_level.tex", out)

# ---------- S9: test-set MC noise sufficiency ----------
rows = [r for r in rd(os.path.join(FULL, "test_set_sufficiency.csv")) if int(r["n_test"]) == 1000]
out = [r"\begin{table}[htbp]", r"\centering",
       r"\caption{\textbf{Supplementary Table S9.} Finite-test-set sensitivity of a pooled-output error diagnostic. An optimised-LHS--GP model is fitted at $20d$ on each of \NumMCNoiseNCases{} cases. At \NumMCNoiseNTest{} test points, the table reports bootstrap-mean $E_{\mathrm{pool}}$ and relative 95\% interval width (width divided by that mean). This scale-weighted diagnostic differs from the primary output-averaged NRMSE. Median and maximum relative widths are \NumMCNoiseMedianRelWidthPct{}\% and \NumMCNoiseMaxRelWidthPct{}\%. Generated from \texttt{test\_set\_sufficiency.csv}.}",
       r"\label{tab:si_mc_noise}", r"\footnotesize", r"\setlength{\tabcolsep}{6pt}",
       r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
       r"\textbf{Case} & \textbf{$n_{\mathrm{test}}$} & \textbf{Mean $E_{\mathrm{pool}}$} & \textbf{Rel.\ CI width} \\",
       r"\midrule"]
for r in sorted(rows, key=lambda r: r["case_id"]):
    out.append(r"%s & %s & %.3f & %.1f\%% \\" % (
        esc(case_label(r["case_id"])), r["n_test"], float(r["nrmse_mean"]), float(r["nrmse_relative_ci_width"]) * 100.0))
out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write("si_mc_noise.tex", out)

# ---------- S3 clean: case ledger ----------
cases = rd(os.path.join(PAPER, "figdata", "fig1_case_catalog.csv"))
tier = {}
for ln in open(os.path.join(TABLES, "si_cases.tex"), encoding="utf-8"):
    s = ln.strip()
    if s.startswith(r"\texttt") and "&" in s:
        f = [x.strip() for x in s.rstrip("\\ ").split("&")]
        nm = f[0].replace(r"\texttt{", "").replace("}", "").replace("\\_", "_")
        tier[nm] = f[-1].replace(r"\texttt{", "").replace("}", "").replace("\\_", "_")
out = [r"\begin{table}[htbp]", r"\centering",
       r"\caption{\textbf{Supplementary Table S3.} Case-characterisation ledger. For each of the \NumNCases{} cases: application domain, input dimension $d$, whether the case carries an explicit failure boundary, and the case-validation evidence tier. Governing equations and source fingerprints are documented in the case sources and \texttt{docs/CASE\_VALIDATION\_SUMMARY.md}. Here $d$ always denotes input dimension; a case identifier suffix such as \texttt{\_2d} may denote physical spatial dimension. Generated from the case registry; repository case identifiers are listed in \texttt{paper/tables/si\_cases.md}.}",
       r"\label{tab:si_cases}", r"\footnotesize", r"\setlength{\tabcolsep}{5pt}",
       r"\begin{tabular}{@{}llccl@{}}", r"\toprule",
       r"\textbf{Case} & \textbf{Domain} & $d$ & \textbf{Failure} & \textbf{Evidence tier} \\",
       r"\midrule"]
for r in sorted(cases, key=lambda r: (r["domain"], int(r["dim"]))):
    fail = "Yes" if r["has_failure"].strip().lower() == "true" else "No"
    out.append(r"%s & %s & %s & %s & %s \\" % (
        esc(case_label(r["case_id"])), domain_label(r["domain"]), r["dim"], fail,
        esc(tier_label(tier.get(r["case_id"], "")))))
out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write("si_cases_clean.tex", out)

# ---------- S2 clean: Sobol with wrapping topology column ----------
raw = [l.rstrip("\n") for l in open(os.path.join(TABLES, "si_sobol.tex"), encoding="utf-8")
       if l.strip().startswith(r"\texttt") and "&" in l]
inter = sum(1 for l in raw if float(l.split("&")[6]) > 1.05)
body = []
for l in raw:
    f = [x.strip() for x in l.rstrip("\\ ").split("&")]
    body.append(r"%s & %s & %s & %s & \texttt{%s} & %s & %s & %s \\" % (
        esc(case_label(unesc(f[0]))), domain_label(unesc(f[1])), f[2],
        topology_label(unesc(f[3])), esc(unesc(f[4])), f[5], f[6], f[7]))
header = (r"\textbf{Case} & \textbf{Domain} & $d$ & \textbf{Topology} & "
          r"\textbf{Top var} & $S_T$ & $\sum S_T$ & \textbf{Inter.} " + r"\\")
out = [r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
       r"\begin{longtable}{@{}>{\raggedright\arraybackslash}p{2.7cm}>{\raggedright\arraybackslash}p{2.1cm}c>{\raggedright\arraybackslash}p{3.6cm}>{\raggedright\arraybackslash}p{1.7cm}rrr@{}}",
       r"\toprule", header, r"\midrule", r"\endfirsthead",
       r"\toprule", header, r"\midrule", r"\endhead"] + body + [r"\bottomrule", r"\end{longtable}"]
write("si_sobol_clean.tex", out)
print("interaction-dominated cases (sumST>1.05):", inter)

# ---------- S1 clean: hyperparameters with wrapping columns ----------
raw = [l.rstrip("\n") for l in open(os.path.join(TABLES, "si_hyperparameters.tex"), encoding="utf-8")
       if l.strip().startswith(r"\texttt") and "&" in l]
body = []
for l in raw:
    f = [x.strip() for x in l.rstrip("\\ ").split("&")]
    body.append(r"%s & %s & %s & %s \\" % (
        esc(method_label(unesc(f[0]))), sampler_label(unesc(f[1])),
        surrogate_label(unesc(f[2])), f[3]))
out = [r"\begin{table}[htbp]", r"\centering",
       r"\captionsetup{justification=raggedright,singlelinecheck=false}",
       r"\caption{\textbf{Supplementary Table S1.} Method pool and locked hyperparameters. Auto-generated from \texttt{benchmark/hyperparameters.py} and guarded against drift by the CI test \texttt{benchmark/\allowbreak tests/\allowbreak test\_hyperparameters.py}. Repository method identifiers for each name are listed in \texttt{paper/tables/si\_hyperparameters.md}.}",
       r"\label{tab:si_hyperparameters}", r"\scriptsize", r"\renewcommand{\arraystretch}{1.0}", r"\setlength{\tabcolsep}{2pt}",
       r"\begin{tabular}{@{}ll>{\raggedright\arraybackslash}p{2.8cm}>{\raggedright\arraybackslash}p{4.8cm}@{}}", r"\toprule",
       r"\textbf{Method} & \textbf{Sampler} & \textbf{Surrogate / kernel} & \textbf{Hyperparameters} \\",
       r"\midrule"] + body + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write("si_hyperparameters_clean.tex", out)
print("done")
