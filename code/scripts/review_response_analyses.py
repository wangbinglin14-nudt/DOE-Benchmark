"""Robustness analyses added at revision (reviewer requests T4, T6, T8, B2).

Outputs (all regenerated, no hand-typed numbers):
  benchmark/results/ranking_reversal/d1_loo_sensitivity.csv
  benchmark/results/reliability_recovery/d3_loco_bca.json
  benchmark/results/full_benchmark/analysis/per_output_nrmse.csv
  paper/si_d1_loo.tex
  paper/si_d1_tau.tex
  paper/si_d3_loco.tex
  paper/si_per_output.tex
  paper/review_response_numbers.json

Run:  python scripts/review_response_analyses.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, norm

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from paper_names import case_label

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.project_layout import paper_file, metadata_file
RR = ROOT / "benchmark" / "results" / "ranking_reversal"
D3 = ROOT / "benchmark" / "results" / "reliability_recovery"
FB = ROOT / "benchmark" / "results" / "full_benchmark"
PAPER = ROOT / "paper"
SEED = 1000
N_BOOT = 10000

PAIRINGS = ["dimension", "topology", "random"]
BUDGETS = [10, 20]


def method_ranks(df: pd.DataFrame, policy: str) -> pd.Series:
    """Mean per-case rank of median NRMSE across cases in df."""
    d = df.copy()
    if policy == "complete_case":
        d = d[d["status"] == "ok"]
        med = d.groupby(["case_id", "method"])["nrmse"].median().unstack()
        # Complete-case means that a physical case contributes only when all
        # methods have a finite score. Ranking the surviving methods in a
        # partially observed case would silently change the registered D1
        # estimand whenever a method timed out.
        med = med.dropna(how="any")
    else:  # penalized: non-ok cells stay in the denominator at worst score
        d.loc[d["status"] != "ok", "nrmse"] = np.inf
        med = d.groupby(["case_id", "method"])["nrmse"].median().unstack()
    return med.rank(axis=1, method="average").mean(axis=0)


def d1_loo() -> tuple[pd.DataFrame, dict]:
    ana = pd.read_csv(RR / "analytical" / "results.csv")
    rows = []
    policy_rhos = {}
    for pairing in PAIRINGS:
        phys = pd.read_csv(RR / f"pairing_{pairing}" / "physical_results.csv")
        for b in BUDGETS:
            ab = ana[ana["budget_mult"] == b]
            pb = phys[phys["budget_mult"] == b]
            for policy in ("complete_case", "penalized"):
                ra_full = method_ranks(ab, policy)
                rp = method_ranks(pb, policy)
                common = ra_full.index.intersection(rp.index)
                rho_full = spearmanr(ra_full[common], rp[common]).statistic
                policy_rhos[f"{pairing}_{b}_{policy}"] = float(rho_full)
                if policy != "complete_case":
                    continue
                for left_out in sorted(ab["case_id"].unique()):
                    ra = method_ranks(ab[ab["case_id"] != left_out], policy)
                    c = ra.index.intersection(rp.index)
                    rho = spearmanr(ra[c], rp[c]).statistic
                    rows.append(
                        dict(pairing=pairing, budget_mult=b, policy=policy,
                             left_out=left_out, rho=float(rho),
                             rho_full=float(rho_full)))
    loo = pd.DataFrame(rows)
    loo.to_csv(RR / "d1_loo_sensitivity.csv", index=False)
    return loo, policy_rhos


def d1_tau() -> list[dict]:
    """Registered D1 rank statistics (Spearman rho + Kendall tau).

    Read from the archived ``correlation.json`` instead of being recomputed:
    the pre-registration names Kendall tau as the required sensitivity
    statistic alongside Spearman rho, and the archive already carries the
    percentile bootstrap interval for both.  Recomputation here would create
    a second, potentially divergent estimate of a registered quantity.
    """
    rows = json.loads((RR / "correlation.json").read_text(encoding="utf-8"))
    rows = [r for r in rows if isinstance(r, dict)]
    order = {p: i for i, p in enumerate(PAIRINGS)}
    rows.sort(key=lambda r: (order.get(r["pairing_mode"], 99),
                             float(r["budget_mult"])))
    return rows


def bca_interval(vals_by_case: dict, alpha=0.05) -> tuple[float, float, float, float]:
    """Case-cluster bootstrap with BCa correction; returns (pct_lo, pct_hi, bca_lo, bca_hi)."""
    cases = sorted(vals_by_case)
    all_vals = np.concatenate([vals_by_case[c] for c in cases])
    theta = all_vals.mean()
    rng = np.random.default_rng(SEED)
    k = len(cases)
    boot = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, k, k)
        s = np.concatenate([vals_by_case[cases[j]] for j in idx])
        boot[i] = s.mean()
    boot.sort()
    pct_lo, pct_hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    # BCa: bias correction + jackknife-over-cases acceleration
    z0 = norm.ppf((boot < theta).mean())
    jack = np.array([
        np.concatenate([vals_by_case[c] for c in cases if c != c0]).mean()
        for c0 in cases])
    jm = jack.mean()
    num = ((jm - jack) ** 3).sum()
    den = 6.0 * (((jm - jack) ** 2).sum() ** 1.5)
    a = num / den if den > 0 else 0.0
    za, zb = norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)
    p_lo = norm.cdf(z0 + (z0 + za) / (1 - a * (z0 + za)))
    p_hi = norm.cdf(z0 + (z0 + zb) / (1 - a * (z0 + zb)))
    bca_lo, bca_hi = np.quantile(boot, [p_lo, p_hi])
    return float(pct_lo), float(pct_hi), float(bca_lo), float(bca_hi), jack, theta


def d3_loco() -> dict:
    df = pd.read_csv(D3 / "reliability_long.csv")
    df = df[(df["budget_mult"] == 10) & df["pf_signed_bias"].notna()]
    by_case = {c: g["pf_signed_bias"].to_numpy() for c, g in df.groupby("case_id")}
    pct_lo, pct_hi, bca_lo, bca_hi, jack, theta = bca_interval(by_case)
    # Registered percentile interval comes from the pre-registered pooled
    # analysis artifact, not from this script's own bootstrap replicate.
    reg = json.loads((D3 / "pooled_reliability_summary.json").read_text(encoding="utf-8"))
    reg = reg[0] if isinstance(reg, list) else reg
    reg_lo = float(reg["cluster_bootstrap_ci_lo"])
    reg_hi = float(reg["cluster_bootstrap_ci_hi"])
    cases = sorted(by_case)
    loco = {c: float(j) for c, j in zip(cases, jack)}
    out = dict(
        n_runs=int(df.shape[0]), n_cases=len(cases),
        pooled_mean_signed_bias=float(theta),
        percentile_ci=[pct_lo, pct_hi],
        registered_percentile_ci=[reg_lo, reg_hi], bca_ci=[bca_lo, bca_hi],
        loco_min=float(jack.min()), loco_max=float(jack.max()),
        loco_min_case=cases[int(jack.argmin())],
        loco_max_case=cases[int(jack.argmax())],
        loco=loco, bootstrap_B=N_BOOT, seed=SEED)
    (D3 / "d3_loco_bca.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def per_output() -> tuple[pd.DataFrame, dict]:
    cols = ["case_id", "status", "budget_mult", "nrmse", "nrmse_out0",
            "nrmse_out1", "nrmse_out2"]
    df = pd.read_csv(FB / "results.csv", usecols=cols)
    df = df[(df["status"] == "ok") & (df["budget_mult"] == 10)]
    g = df.groupby("case_id")[["nrmse", "nrmse_out0", "nrmse_out1", "nrmse_out2"]].median()
    g["worst_out"] = g[["nrmse_out0", "nrmse_out1", "nrmse_out2"]].max(axis=1)
    g["best_out"] = g[["nrmse_out0", "nrmse_out1", "nrmse_out2"]].min(axis=1)
    g["spread"] = g["worst_out"] - g["best_out"]
    g["worst_minus_mean"] = g["worst_out"] - g["nrmse"]
    g = g.round(4)
    g.to_csv(FB / "analysis" / "per_output_nrmse.csv")
    summ = dict(
        n_cases=int(g.shape[0]),
        median_spread=float(g["spread"].median()),
        max_spread=float(g["spread"].max()),
        max_spread_case=str(g["spread"].idxmax()),
        n_worst_exceeds_mean_by_010=int((g["worst_minus_mean"] > 0.10).sum()),
        n_worst_exceeds_mean_by_020=int((g["worst_minus_mean"] > 0.20).sum()))
    return g, summ


def write_tex(loo: pd.DataFrame, policy_rhos: dict, d3: dict, po: pd.DataFrame,
              po_summ: dict, tau: list[dict]) -> None:
    # --- SI: D1 leave-one-out ---
    prim = loo[(loo.pairing == "dimension")]
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{\textbf{Supplementary Table S10.} Leave-one-function-out"
        r" sensitivity of the D1 rank-transfer correlation. Each row removes one"
        r" of the six analytical anchor functions and recomputes Spearman"
        r" $\rho$ between the mean analytical and physical method ranks under"
        r" the registered \texttt{complete\_case} policy (dimension-matched"
        r" pairing). All leave-one-out values remain above the registered"
        r" $\rho<0.70$ rank-reversal threshold."
        r" Generated from \texttt{d1\_loo\_sensitivity.csv}.}",
        r"\label{tab:si_d1_loo}", r"\footnotesize",
        r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
        r"\textbf{Function left out} & $\rho$ ($10d$) & $\rho$ ($20d$) \\",
        r"\midrule"]
    for lo in sorted(prim.left_out.unique()):
        r10 = prim[(prim.left_out == lo) & (prim.budget_mult == 10)].rho.iloc[0]
        r20 = prim[(prim.left_out == lo) & (prim.budget_mult == 20)].rho.iloc[0]
        lines.append(rf"{case_label(lo)} & {r10:.3f} & {r20:.3f} \\")
    f10 = prim[prim.budget_mult == 10].rho_full.iloc[0]
    f20 = prim[prim.budget_mult == 20].rho_full.iloc[0]
    lines += [r"\midrule",
              rf"None (all six functions) & {f10:.3f} & {f20:.3f} \\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (paper_file(PAPER, "si_d1_loo.tex")).write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- SI: D3 leave-one-case-out + BCa ---
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{\textbf{Supplementary Table S11.} Robustness of the pooled"
        r" D3 signed bias. Top: pooled estimate with percentile and BCa"
        r" case-cluster bootstrap intervals. Bottom: range of the pooled"
        r" estimate when each of the " + str(d3["n_cases"]) +
        r" failure-boundary cases is left out in turn (leave-one-case-out);"
        r" no single case drives the positive bias."
        r" Generated from \texttt{pooled\_reliability\_summary.json}"
        r" (registered percentile interval) and \texttt{d3\_loco\_bca.json}.}",
        r"\label{tab:si_d3_loco}", r"\footnotesize",
        r"\begin{tabular}{@{}lc@{}}", r"\toprule",
        r"\textbf{Quantity} & \textbf{Value} \\", r"\midrule",
        rf"Pooled mean signed bias & {d3['pooled_mean_signed_bias']:.4f} \\",
        rf"95\% percentile cluster-bootstrap CI (registered) & [{d3['registered_percentile_ci'][0]:.4f}, {d3['registered_percentile_ci'][1]:.4f}] \\",
        rf"95\% BCa cluster-bootstrap CI & [{d3['bca_ci'][0]:.4f}, {d3['bca_ci'][1]:.4f}] \\",
        rf"Leave-one-case-out minimum & {d3['loco_min']:.4f} ({case_label(d3['loco_min_case'])}) \\",
        rf"Leave-one-case-out maximum & {d3['loco_max']:.4f} ({case_label(d3['loco_max_case'])}) \\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (paper_file(PAPER, "si_d3_loco.tex")).write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- SI: per-output NRMSE ---
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{\textbf{Supplementary Table S12.} Per-output NRMSE at the"
        r" $10d$ budget. For each case: the median (across methods and seeds)"
        r" three-output-mean NRMSE, the per-output medians, and the"
        r" worst-minus-best output spread. Generated from"
        r" \texttt{per\_output\_nrmse.csv}.}",
        r"\label{tab:si_per_output}", r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
        r"\textbf{Case} & \textbf{Mean} & \textbf{Out 1} & \textbf{Out 2} & \textbf{Out 3} & \textbf{Spread} \\",
        r"\midrule"]
    for case, row in po.sort_index().iterrows():
        lines.append(
            rf"{case_label(case)} & {row['nrmse']:.3f} & "
            rf"{row['nrmse_out0']:.3f} & {row['nrmse_out1']:.3f} & "
            rf"{row['nrmse_out2']:.3f} & {row['spread']:.3f} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (paper_file(PAPER, "si_per_output.tex")).write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- SI: registered D1 rank statistics (Spearman rho + Kendall tau) ---
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{\textbf{Supplementary Table S13.} Registered D1 rank"
        r" statistics. The pre-registration names Spearman $\rho$ as the primary"
        r" D1 statistic and Kendall $\tau$ as the required sensitivity statistic,"
        r" which must agree with it in sign and order. Both are computed on the"
        r" $N=20$ method ranks for every registered pairing and budget, each with"
        r" a percentile bootstrap 95\% interval over the method axis"
        r" ($B=10^{4}$). Generated from \texttt{correlation.json}.}",
        r"\label{tab:si_d1_tau}", r"\footnotesize",
        r"\begin{tabular}{@{}llcc@{}}", r"\toprule",
        r"\textbf{Pairing} & \textbf{Budget} & $\rho$ [95\% CI] & $\tau$ [95\% CI] \\",
        r"\midrule"]
    for r in tau:
        lines.append(
            rf"{r['pairing_mode']} & {int(r['budget_mult'])}d"
            rf" & {r['spearman_rho']:.3f} [{r['spearman_rho_ci_lo']:.3f}, {r['spearman_rho_ci_hi']:.3f}]"
            rf" & {r['kendall_tau']:.3f} [{r['kendall_tau_ci_lo']:.3f}, {r['kendall_tau_ci_hi']:.3f}] \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (paper_file(PAPER, "si_d1_tau.tex")).write_text("\n".join(lines) + "\n", encoding="utf-8")

    nums = dict(
        d1_loo_min_rho_10d=float(prim[prim.budget_mult == 10].rho.min()),
        d1_loo_min_rho_20d=float(prim[prim.budget_mult == 20].rho.min()),
        d1_loo_min_rho_all_pairings=float(loo.rho.min()),
        d1_policy_rhos=policy_rhos,
        d3=dict((k, v) for k, v in d3.items() if k != "loco"),
        per_output=po_summ,
        d1_rank_statistics=[
            dict(pairing=r["pairing_mode"], budget=int(r["budget_mult"]),
                 rho=float(r["spearman_rho"]), tau=float(r["kendall_tau"]))
            for r in tau])
    (PAPER / "review_response_numbers.json").write_text(
        json.dumps(nums, indent=2), encoding="utf-8")
    print(json.dumps(nums, indent=2))


def main() -> None:
    loo, policy_rhos = d1_loo()
    d3 = d3_loco()
    po, po_summ = per_output()
    tau = d1_tau()
    write_tex(loo, policy_rhos, d3, po, po_summ, tau)


if __name__ == "__main__":
    main()
