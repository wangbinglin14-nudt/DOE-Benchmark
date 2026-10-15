"""
benchmark.scripts.generate_report
===================================
Single-shot rigorous report generator for any benchmark CSV.

Given a directory containing ``results.csv`` (e.g.
``benchmark/results/extended_pilot``) this script produces:

  - All standard analyses (CD diagram, perf heatmap, ranks)
  - Stratified ranks (VS scenario, topology, domain, dim bucket)
  - Bootstrap rank CI forest plot
  - Pairwise win-rate matrix + dominance score
  - Cliff's delta effect-size matrix
  - Compute-vs-accuracy Pareto front
  - Per-output NRMSE rank-shift diagnostic
  - Method robustness (CV across reps)
  - Cross-metric ranking robustness
  - Fit-quality flags
  - REPORT.md auto-generated with all the above embedded

Usage
-----
    python benchmark/scripts/generate_report.py benchmark/results/extended_pilot
"""

from __future__ import annotations

import io
import sys
import warnings
from pathlib import Path
from textwrap import dedent

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark.analysis import critical_difference, friedman_test, pivot_metric, rank_methods
from benchmark.dim_scaling import (
    alpha_vs_dim,
    default_dim_bands,
    nrmse_vs_dim,
    rank_by_dimension_band,
)
from benchmark.learning_curves import (
    fit_power_law,
    learning_curve_table,
    rank_stability_score,
    ranking_at_each_budget,
)
from benchmark.plots import (
    plot_cd_diagram,
    plot_multi_stratum_grid,
    plot_pairwise_win_matrix,
    plot_pareto,
    plot_performance_heatmap,
    plot_rank_ci,
    plot_winners_per_case,
    use_paper_style,
)
from benchmark.rigour import (
    cliffs_delta_ci_matrix,
    cliffs_delta_matrix,
    dominance_score,
    failure_rate_per_method,
    fit_quality_flags,
    pairwise_signed_rank_test,
    pairwise_win_rate_matrix,
    pareto_front,
    per_output_rank_shifts,
    per_output_summary,
)
from benchmark.stratified import (
    annotate_results,
    bootstrap_rank_ci,
    bootstrap_rank_shift_ci,
    stratified_ranks,
    winner_table,
)


# ─── Markdown helpers ────────────────────────────────────────────────────
def md_table(df: pd.DataFrame, max_rows: int = 20, float_fmt: str = "{:.3f}") -> str:
    """Render a DataFrame as a markdown table (truncated to max_rows)."""
    if df is None or len(df) == 0:
        return "_(no data)_\n"
    df = df.head(max_rows)
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)) and np.isfinite(v):
                cells.append(float_fmt.format(v))
            elif pd.isna(v):
                cells.append("—")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def md_section(title: str, level: int = 2) -> str:
    return f"\n{'#' * level} {title}\n\n"


def sensitivity_paths(results_dir: Path) -> tuple[Path, Path]:
    """Return sensitivity-recovery files scoped to this results archive.

    Keep supplementary analyses under the same run directory as
    ``results.csv``.  Reading from ``results_dir.parent`` can silently mix a
    smoke or pilot report with stale sibling outputs from a different run.
    """
    base = results_dir / "sensitivity_recovery"
    return base / "sensitivity_summary.csv", base / "sensitivity_recovery.csv"


def failure_accounting_section(raw_df: pd.DataFrame) -> str:
    """Markdown status/failure accounting for the unfiltered result archive."""
    if "status" not in raw_df.columns:
        return dedent("""

        ## Failure accounting

        This archive has no `status` column, so the report treats it as a
        legacy all-success CSV. Formal headline archives must include row-level
        `status` and `error` fields.

        """)

    df = raw_df.copy()
    df["status"] = df["status"].fillna("<missing>").astype(str).str.strip()
    n_total = len(df)
    status_counts = (
        df.groupby("status", dropna=False)
        .size()
        .rename("n_rows")
        .reset_index()
        .sort_values(["n_rows", "status"], ascending=[False, True])
    )
    if n_total:
        status_counts["fraction"] = status_counts["n_rows"] / n_total
    else:
        status_counts["fraction"] = np.nan

    non_ok = df[df["status"] != "ok"].copy()
    md = [md_section("Failure accounting", level=2)]
    md.append(dedent("""
    Downstream ranking sections use only `status == "ok"` rows for
    complete-case accuracy comparisons. This section reports the unfiltered
    row-status distribution so failed or timed-out cells remain visible in the
    generated archive.

    """))
    md.append("**Status counts:**\n\n")
    md.append(md_table(status_counts, max_rows=20))
    if "failure_category" in df.columns:
        df["failure_category"] = (
            df["failure_category"].fillna("<missing>").astype(str).str.strip()
        )
        category_counts = (
            df.groupby("failure_category", dropna=False)
            .size()
            .rename("n_rows")
            .reset_index()
            .sort_values(["n_rows", "failure_category"], ascending=[False, True])
        )
        if n_total:
            category_counts["fraction"] = category_counts["n_rows"] / n_total
        else:
            category_counts["fraction"] = np.nan
        md.append("\n**Failure-category counts:**\n\n")
        md.append(md_table(category_counts, max_rows=20))

    if non_ok.empty:
        md.append("\nNo failed, timeout, missing-status, or otherwise non-ok rows were present.\n")
        return "".join(md)

    def _failure_rate_table(group_col: str) -> pd.DataFrame:
        if group_col not in df.columns:
            return pd.DataFrame()
        df["_row_count"] = 1
        counts = (
            df.pivot_table(
                index=group_col,
                columns="status",
                values="_row_count",
                aggfunc="count",
                fill_value=0,
            )
            .reset_index()
        )
        status_cols = [c for c in counts.columns if c != group_col]
        counts["n_total"] = counts[status_cols].sum(axis=1)
        counts["n_non_ok"] = counts["n_total"] - counts.get("ok", 0)
        counts["non_ok_rate"] = counts["n_non_ok"] / counts["n_total"].replace(0, np.nan)
        preferred = [group_col, "n_total", "n_non_ok", "non_ok_rate", "failed", "timeout", "<missing>"]
        ordered = [c for c in preferred if c in counts.columns]
        rest = [c for c in counts.columns if c not in ordered and c not in {"ok"}]
        return (
            counts[ordered + rest]
            .sort_values(["non_ok_rate", "n_non_ok", group_col], ascending=[False, False, True])
            .reset_index(drop=True)
        )

    by_method = _failure_rate_table("method")
    if not by_method.empty:
        md.append("\n**Highest non-ok rates by method:**\n\n")
        md.append(md_table(by_method, max_rows=20))

    by_case = _failure_rate_table("case_id")
    if not by_case.empty:
        md.append("\n**Highest non-ok rates by case:**\n\n")
        md.append(md_table(by_case, max_rows=20))

    example_cols = [
        c
        for c in [
            "case_id",
            "method",
            "budget_mult",
            "repeat",
            "status",
            "failure_category",
            "error",
        ]
        if c in non_ok.columns
    ]
    if example_cols:
        examples = non_ok[example_cols].copy()
        if "error" in examples.columns:
            examples["error"] = examples["error"].fillna("").astype(str).str.slice(0, 160)
        md.append("\n**First non-ok row examples:**\n\n")
        md.append(md_table(examples.head(10), max_rows=10))

    return "".join(md)


# ─── Main report generator ───────────────────────────────────────────────
def generate(results_dir: Path) -> None:
    results_dir = results_dir.resolve()
    csv = results_dir / "results.csv"
    try:
        csv_display = csv.relative_to(ROOT)
    except ValueError:
        csv_display = csv
    raw_df = pd.read_csv(csv)
    n_rows_total = len(raw_df)
    df = raw_df
    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()
    n_rows_ok = len(df)
    df = annotate_results(df)

    out = results_dir / "report"
    out.mkdir(exist_ok=True)
    figs = out / "figs"
    figs.mkdir(exist_ok=True)
    use_paper_style()

    largest = int(max(df["budget_mult"].unique()))
    smallest = int(min(df["budget_mult"].unique()))
    n_cases = df["case_id"].nunique()
    n_methods = df["method"].nunique()
    n_repeats = df["repeat"].nunique()

    md = []
    md.append(f"# Benchmark Report: `{results_dir.name}`\n\n")
    md.append(dedent(f"""
    Auto-generated by `benchmark.scripts.generate_report` on the
    contents of `{csv_display}`.

    **Coverage**

    - cases   : {n_cases}
    - methods : {n_methods}
    - budgets : {sorted(int(b) for b in df['budget_mult'].unique())} (multiples of d)
    - repeats : {n_repeats}
    - rows    : {n_rows_ok} successful out of {n_rows_total} ({n_rows_ok/n_rows_total*100:.1f}%)

    """))
    md.append(failure_accounting_section(raw_df))

    # ─── Executive Summary (top-of-report TL;DR) ─────────────────────────
    # Compute the headline figures once so we can distill them into bullets.
    pivot_exec = pivot_metric(df, "nrmse", budget_value=largest)
    ranks_exec = rank_methods(pivot_exec, lower_is_better=True)
    fstat_exec, fp_exec = friedman_test(pivot_exec)
    cd_exec = critical_difference(len(ranks_exec), pivot_exec.dropna().shape[0])
    top1 = ranks_exec.index[0]
    top1_rank = float(ranks_exec.iloc[0])
    second = ranks_exec.index[1] if len(ranks_exec) > 1 else None
    bottom = ranks_exec.index[-1]
    bottom_rank = float(ranks_exec.iloc[-1])

    # Pareto analysis (compute optimal vs accuracy)
    try:
        pa_exec = pareto_front(df, metric="nrmse", cost_col="t_total_s",
                               budget_value=largest)
        pareto_methods = pa_exec[pa_exec["on_pareto"]]["method"].tolist()
    except Exception:
        pa_exec = None
        pareto_methods = []

    md.append(md_section("Executive Summary", level=2))
    md.append(f"**Headline finding** (NRMSE @ {largest}·d on {pivot_exec.dropna().shape[0]} cases, {len(ranks_exec)} methods):\n\n"
              )
    md.append(f"- Friedman χ² = {fstat_exec:.1f}, p = {fp_exec:.1e}; "
              f"Nemenyi CD (α=0.05) = {cd_exec:.2f}.\n")
    md.append(f"- Best mean rank: **`{top1}`** (rank {top1_rank:.2f})"
              + (f", followed by `{second}`.\n" if second else ".\n"))
    md.append(f"- Worst mean rank: `{bottom}` (rank {bottom_rank:.2f}).\n")
    if pareto_methods:
        md.append(f"- Pareto-optimal compute-vs-accuracy: "
                  f"{', '.join(f'`{m}`' for m in pareto_methods)}.\n")
    # Per-output spread highlight — works whenever per-output NRMSE columns exist
    if any(c.startswith("nrmse_out") for c in df.columns):
        try:
            pos_exec = per_output_summary(df, budget_value=largest)
            if not pos_exec.empty and "mean_spread" in pos_exec.columns:
                worst_imbalance = pos_exec.loc[pos_exec["mean_spread"].idxmax()]
                md.append(f"- Largest per-output imbalance: "
                          f"`{worst_imbalance['method']}` with "
                          f"spread = {worst_imbalance['mean_spread']:.3f} "
                          f"(out0/out1/out2 NRMSE differ wildly — see §7).\n")
        except Exception:
            pass

    # Cross-metric robustness (one-shot)
    md.append("\n**Robustness checks**:\n\n")
    try:
        from scipy.stats import spearmanr as _sp
        candidate_metrics = ["nrmse", "rmse", "mae", "maxae", "q95ae", "r2"]
        rd = {}
        df_b = df[df["budget_mult"] == largest]
        for m in candidate_metrics:
            if m in df_b.columns:
                pv = pivot_metric(df_b, m)
                rd[m] = rank_methods(pv,
                                     lower_is_better=(m != "r2"))
        if len(rd) >= 2:
            metrics = list(rd.keys())
            min_rho = 1.0
            for m1 in metrics:
                for m2 in metrics:
                    if m1 == m2:
                        continue
                    rho, _ = _sp(rd[m1], rd[m2])
                    if rho < min_rho:
                        min_rho = rho
            md.append(f"- Cross-metric ranking consistency: "
                      f"min Spearman ρ = **{min_rho:.3f}** "
                      f"(headline holds under all of "
                      f"{', '.join(metrics)}).\n")
    except Exception:
        pass

    md.append(f"- Sample size: {pivot_exec.dropna().shape[0]} cases × "
              f"{n_repeats} repeats; bootstrap CI in §2.\n")

    # Holm-Bonferroni quick stat
    try:
        psr_exec = pairwise_signed_rank_test(df, metric="nrmse",
                                              budget_value=largest, alpha=0.05)
        if psr_exec and not psr_exec["summary"].empty:
            n_total = len(psr_exec["summary"])
            n_sig = int(psr_exec["summary"]["significant_holm"].sum())
            md.append(f"- Holm-Bonferroni post-hoc (Wilcoxon, α=0.05): "
                      f"**{n_sig} / {n_total} pairs** "
                      f"({100 * n_sig / n_total:.0f} %) significantly "
                      f"different; details in §3b.\n")
    except Exception:
        pass

    # Dimensional scaling highlight (if df has ≥3 distinct dims)
    if "dim" in df.columns and df["dim"].nunique() >= 3:
        try:
            rk_dim_exec = rank_by_dimension_band(df, budget_value=largest)
            nv_exec = nrmse_vs_dim(df, budget_value=largest)
            if not rk_dim_exec.empty:
                top_mover = rk_dim_exec["range"].idxmax()
                top_range = float(rk_dim_exec.loc[top_mover, "range"])
                if top_range >= 2:
                    md.append(f"- **Dimensional scaling**: "
                              f"`{top_mover}` rank shifts by "
                              f"{top_range:.1f} positions across "
                              f"dim bands — see §7c.\n")
            if not nv_exec.empty:
                worst_curse = nv_exec.iloc[-1]
                if float(worst_curse["log_slope"]) > 0.05:
                    md.append(f"- **Curse-of-dim severity**: "
                              f"`{worst_curse['method']}` has "
                              f"log-NRMSE-vs-dim slope = "
                              f"{float(worst_curse['log_slope']):.3f} "
                              f"(R² = {float(worst_curse['R2']):.2f}) — "
                              f"degrades fastest with dimension.\n")
        except Exception:
            pass

    # ─── Sensitivity-recovery vs NRMSE decoupling (executive bullet) ─────
    sr_summary_path, _ = sensitivity_paths(results_dir)
    if sr_summary_path.exists():
        try:
            from scipy.stats import spearmanr as _sp_sr
            sr_summary = pd.read_csv(sr_summary_path)
            if "method" in sr_summary.columns and "tau_mean" in sr_summary.columns:
                sr_indexed = sr_summary.set_index("method")["tau_mean"]
                # Best by SR
                best_sr = sr_indexed.idxmax()
                best_sr_tau = float(sr_indexed.max())
                # Spearman correlation between SR rank and NRMSE rank
                common = sr_indexed.index.intersection(ranks_exec.index)
                if len(common) >= 4:
                    sr_rank = (-sr_indexed.loc[common]).rank()  # higher τ = better
                    nr_rank = ranks_exec.loc[common]
                    rho_sr, p_sr = _sp_sr(sr_rank, nr_rank)
                    md.append(f"- **Sensitivity recovery**: best method = "
                              f"`{best_sr}` (mean Kendall τ = "
                              f"{best_sr_tau:.3f} vs ground-truth Sobol-ST). "
                              f"Spearman ρ between NRMSE rank and SR rank "
                              f"= **{rho_sr:+.3f}** "
                              f"({'aligned' if rho_sr > 0.7 else 'partially decoupled' if rho_sr > 0.3 else 'decoupled'} "
                              f"— see §10b).\n")
        except Exception:
            pass

    # ─── 1. Headline ──────────────────────────────────────────────────────
    md.append(md_section("1. Headline ranking"))
    pivot = pivot_metric(df, "nrmse", budget_value=largest)
    ranks = rank_methods(pivot, lower_is_better=True)
    fstat, fp = friedman_test(pivot)
    n_methods_h, n_cases_h = len(ranks), pivot.dropna().shape[0]
    cd = critical_difference(n_methods_h, n_cases_h)

    md.append(dedent(f"""
    Headline metric: NRMSE on the largest budget ({largest}·d) over the {n_cases_h} cases
    where every method completed.

    - Friedman χ² = **{fstat:.2f}**, p = **{fp:.2e}**
    - Critical difference (Nemenyi α = 0.05) = **{cd:.3f}**
    - Methods within {cd:.2f} mean-rank units are not statistically separable.

    """))
    rank_table = ranks.reset_index()
    rank_table.columns = ["method", "mean_rank"]
    md.append(md_table(rank_table, max_rows=n_methods_h))

    # CD diagram
    fig, ax = plt.subplots(figsize=(11, 0.5 + 0.3 * n_methods_h))
    plot_cd_diagram(pivot, lower_is_better=True,
                    title=f"NRMSE @ {largest}·d (N={n_cases_h}, k={n_methods_h})",
                    ax=ax)
    fig.tight_layout()
    fig.savefig(figs / "01_cd_diagram.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    md.append("\n![CD diagram](figs/01_cd_diagram.png)\n")

    # ─── 1b. Budget-induced rank shifts with bootstrap CI ───────────────
    budgets_sorted = sorted(df["budget_mult"].unique())
    if len(budgets_sorted) >= 2:
        smallest, largest_b = budgets_sorted[0], budgets_sorted[-1]
        md.append(md_section(
            f"1b. Budget-induced rank shifts ({smallest}·d → {largest_b}·d)",
            level=2,
        ))
        md.append(dedent(f"""
        How does each method's rank change as the training budget grows
        from `{smallest}·d` to `{largest_b}·d`?  We resample cases with
        replacement (n_boot = 1000) to obtain a 95 % bootstrap CI on
        the rank shift.  A method whose CI excludes zero has a
        statistically robust rank change between the two budgets.

        - **Negative shift** = method *improves* (climbs the ranking)
          when the budget grows.
        - **Positive shift** = method *degrades* relative to peers as
          the budget grows.
        - `shift_significant = True` ⇒ rank change is robust to case
          resampling at α = 0.05 (one-sided).

        """))
        try:
            # The helper accepts two case populations; here we filter
            # the same df by budget so the *populations* are identical
            # but the underlying NRMSE distributions differ.
            df_small = df[df["budget_mult"] == smallest]
            df_large = df[df["budget_mult"] == largest_b]
            shift_df = bootstrap_rank_shift_ci(
                df_a=df_small, df_b=df_large,
                metric="nrmse",
                lower_is_better=True,
                budget_value=None,
                n_boot=1000,
                seed=0,
            )
            n_sig = int(shift_df["shift_significant"].sum())
            md.append(
                f"\n**{n_sig} / {len(shift_df)} methods have a "
                f"statistically robust rank shift between the two "
                f"budgets at α = 0.05.**\n\n"
            )
            md.append(md_table(shift_df, max_rows=len(shift_df)))

            fig, ax = plt.subplots(figsize=(9, 0.4 * len(shift_df) + 1.2))
            order = shift_df.sort_values("shift")
            y = np.arange(len(order))
            colors = ["#d62728" if s > 0 else "#2ca02c"
                       for s in order["shift"]]
            ax.barh(y, order["shift"], color=colors, alpha=0.7,
                     xerr=[order["shift"] - order["shift_ci_low"],
                           order["shift_ci_high"] - order["shift"]],
                     ecolor="black", capsize=2)
            ax.axvline(0, color="black", lw=0.6)
            ax.set_yticks(y)
            ax.set_yticklabels(order["method"], fontsize=8)
            ax.set_xlabel(f"Rank shift  ({smallest}·d → {largest_b}·d) "
                          f"with 95 % bootstrap CI")
            ax.set_title("Budget-induced rank shifts")
            ax.invert_yaxis()
            fig.tight_layout()
            fig.savefig(figs / "01b_budget_rank_shifts.png",
                        dpi=300, bbox_inches="tight")
            plt.close(fig)
            md.append("\n![Budget rank shifts]"
                      "(figs/01b_budget_rank_shifts.png)\n")
        except Exception as exc:
            md.append(f"\n_(budget rank-shift CI failed: {exc})_\n")

    # Bootstrap rank CI
    md.append(md_section("2. Bootstrap rank CI", level=2))
    md.append(dedent("""
    Each method's mean rank with a 95 % bootstrap CI obtained by
    resampling cases with replacement (n_boot=1000).  Methods with
    overlapping CIs are not statistically distinguishable on this
    sample of cases.

    """))
    ci = bootstrap_rank_ci(df, budget_value=largest, n_boot=1000, seed=0)
    md.append(md_table(ci, max_rows=n_methods_h))

    fig, ax = plt.subplots(figsize=(9, 0.4 * len(ci) + 1.2))
    plot_rank_ci(ci, ax=ax,
                 title="Mean rank with bootstrap 95% CI (n_boot=1000)")
    fig.tight_layout()
    fig.savefig(figs / "02_rank_ci_forest.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    md.append("\n![Rank CI forest](figs/02_rank_ci_forest.png)\n")

    # ─── 3. Pairwise win-rate matrix ─────────────────────────────────────
    md.append(md_section("3. Pairwise win-rate matrix", level=2))
    md.append(dedent("""
    Each cell `(i, j)` = fraction of (case, repeat) pairs in which method
    `i` has lower NRMSE than method `j` (paired comparison on the same
    test set).  Diagonal omitted.  Rows are sorted by *dominance score*
    = mean win rate against all other methods.

    """))
    win_mat = pairwise_win_rate_matrix(df, metric="nrmse",
                                       budget_value=largest)
    dom = dominance_score(win_mat).sort_values(ascending=False)
    md.append("\n**Dominance score (mean win rate against all opponents):**\n\n")
    md.append(md_table(dom.reset_index().rename(
        columns={"index": "method", 0: "dominance"})))

    fig, ax = plt.subplots(figsize=(0.55 * len(win_mat) + 2, 0.55 * len(win_mat) + 1))
    plot_pairwise_win_matrix(win_mat, ax=ax)
    ax.set_title(f"Pairwise win-rate (NRMSE @ {largest}·d)")
    fig.tight_layout()
    fig.savefig(figs / "03_pairwise_win_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    md.append("\n![Win-rate matrix](figs/03_pairwise_win_matrix.png)\n")

    # ─── 3b. Holm-Bonferroni pairwise signed-rank tests ──────────────────
    md.append(md_section("3b. Pairwise Holm-Bonferroni significance test",
                         level=2))
    md.append(dedent("""
    For every method pair we run a paired Wilcoxon signed-rank test on
    per-case mean NRMSE and apply the Holm-Bonferroni procedure to
    control family-wise error rate at α=0.05.  Compared to the Nemenyi
    CD test (§1), this is a *more powerful* post-hoc — it uses
    paired-difference information instead of average ranks.

    """))
    try:
        psr = pairwise_signed_rank_test(df, metric="nrmse",
                                         budget_value=largest, alpha=0.05)
        if psr and not psr["summary"].empty:
            n_total = len(psr["summary"])
            n_sig = int(psr["summary"]["significant_holm"].sum())
            md.append(f"\n**{n_sig} / {n_total} method pairs "
                      f"({100 * n_sig / n_total:.0f} %) are significantly "
                      f"different at family-wise α=0.05.**\n\n")

            top10 = psr["summary"].head(10)[
                ["method_A", "method_B", "n_cases",
                 "mean_diff", "p_raw", "p_holm"]
            ]
            md.append("\n**Top-10 most significant pairs:**\n\n")
            md.append(md_table(top10, max_rows=10))

            # Pairs that *cannot* be distinguished — practical equivalence
            ns = psr["summary"][~psr["summary"]["significant_holm"]]
            if not ns.empty:
                ns_disp = ns.tail(8)[
                    ["method_A", "method_B", "n_cases",
                     "mean_diff", "p_holm"]
                ]
                md.append("\n**Equivalent (non-significant) pairs — "
                          "method swaps with no measurable effect:**\n\n")
                md.append(md_table(ns_disp, max_rows=8))

            # Plot the significant matrix
            sig_mat = psr["significant"].astype(int)
            method_order = list(ranks.index)
            sig_mat = sig_mat.reindex(index=method_order, columns=method_order)
            fig, ax = plt.subplots(figsize=(0.4 * len(method_order) + 2,
                                             0.4 * len(method_order) + 2))
            im = ax.imshow(sig_mat.to_numpy(dtype=float),
                           cmap="Greys", vmin=0, vmax=1, aspect="equal")
            ax.set_xticks(range(len(method_order)))
            ax.set_xticklabels(method_order, rotation=45, ha="right",
                               fontsize=7)
            ax.set_yticks(range(len(method_order)))
            ax.set_yticklabels(method_order, fontsize=7)
            ax.set_title(f"Holm-significant pairs at α=0.05  "
                         f"(NRMSE @ {largest}·d)")
            plt.colorbar(im, ax=ax, fraction=0.04,
                         label="1 = significantly different")
            fig.tight_layout()
            fig.savefig(figs / "03b_holm_significance.png",
                        dpi=300, bbox_inches="tight")
            plt.close(fig)
            md.append("\n![Holm significance](figs/03b_holm_significance.png)\n")
    except Exception as exc:
        md.append(f"\n_(Holm test failed: {exc})_\n")

    # ─── 4. Cliff's delta effect-size with bootstrap CI ─────────────────
    md.append(md_section("4. Cliff's delta effect-size matrix (with 95 % CI)", level=2))
    md.append(dedent("""
    Cliff's δ is a non-parametric effect-size measure.

    - |δ| < 0.147 : negligible
    - 0.147–0.33  : small
    - 0.33–0.474  : medium
    - ≥ 0.474     : large

    A method `i` "soundly beats" `j` only when δ(i, j) is medium or large
    **and** the 95 % bootstrap CI on δ excludes 0.  We hatch cells whose
    CI overlaps zero so they are visually distinguishable from cells
    that are simultaneously large *and* statistically robust.
    """))
    try:
        cd_ci = cliffs_delta_ci_matrix(
            df, metric="nrmse", budget_value=largest,
            n_boot=500, seed=0,
        )
        delta_mat = cd_ci["delta"]
        non_neg = cd_ci["non_negligible"].to_numpy()
        n_pairs = int(non_neg.sum() / 2)  # symmetry: count each pair once
        md.append(
            f"\n**{n_pairs} ordered pairs of methods are non-negligible "
            f"(|δ| ≥ 0.147 with CI excluding 0) at the {largest}·d "
            f"budget.**\n\n"
        )

        fig, ax = plt.subplots(figsize=(0.55 * len(delta_mat) + 2,
                                         0.55 * len(delta_mat) + 1))
        im = ax.imshow(delta_mat.to_numpy(dtype=float), cmap="RdBu_r",
                       vmin=-1, vmax=1)
        # Hatch cells that are NOT statistically distinguishable
        for i in range(len(delta_mat)):
            for j in range(len(delta_mat)):
                if i == j:
                    continue
                if not non_neg[i, j]:
                    ax.add_patch(plt.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1, fill=False,
                        edgecolor="lightgray", hatch="////",
                        linewidth=0,
                    ))
        ax.set_xticks(np.arange(len(delta_mat)))
        ax.set_xticklabels(delta_mat.columns, rotation=45, ha="right",
                           fontsize=8)
        ax.set_yticks(np.arange(len(delta_mat)))
        ax.set_yticklabels(delta_mat.index, fontsize=8)
        ax.set_title(
            f"Cliff's δ (row vs col), NRMSE @ {largest}·d  (hatched = CI overlaps 0)"
            
        )
        plt.colorbar(im, ax=ax, fraction=0.04)
        fig.tight_layout()
        fig.savefig(figs / "04_cliffs_delta.png", dpi=300,
                    bbox_inches="tight")
        plt.close(fig)
        md.append("\n![Cliff's delta](figs/04_cliffs_delta.png)\n")
    except Exception as exc:
        md.append(f"\n_(Cliff's δ CI failed: {exc}; falling back to "
                  "point estimate)_\n")
        delta_mat = cliffs_delta_matrix(df, metric="nrmse",
                                        budget_value=largest)
        fig, ax = plt.subplots(figsize=(0.55 * len(delta_mat) + 2,
                                         0.55 * len(delta_mat) + 1))
        im = ax.imshow(delta_mat.to_numpy(dtype=float), cmap="RdBu_r",
                       vmin=-1, vmax=1)
        ax.set_xticks(np.arange(len(delta_mat)))
        ax.set_xticklabels(delta_mat.columns, rotation=45, ha="right",
                           fontsize=8)
        ax.set_yticks(np.arange(len(delta_mat)))
        ax.set_yticklabels(delta_mat.index, fontsize=8)
        ax.set_title(f"Cliff's δ (row vs col), NRMSE @ {largest}·d")
        plt.colorbar(im, ax=ax, fraction=0.04)
        fig.tight_layout()
        fig.savefig(figs / "04_cliffs_delta.png", dpi=300,
                    bbox_inches="tight")
        plt.close(fig)
        md.append("\n![Cliff's delta](figs/04_cliffs_delta.png)\n")

    # ─── 5. Pareto front (compute vs accuracy) ────────────────────────────
    md.append(md_section("5. Compute-vs-accuracy Pareto front", level=2))
    md.append(dedent("""
    The Pareto-optimal frontier identifies methods that are not
    strictly dominated on both axes (cost and accuracy).  Methods on
    the front are the only sensible compute-budget choices; methods
    below it are dominated and should not be recommended for that
    budget.

    """))
    pa = pareto_front(df, metric="nrmse", cost_col="t_total_s",
                      budget_value=largest)
    md.append(md_table(pa.head(20)))

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_pareto(pa, ax=ax,
                title=f"Compute-vs-NRMSE Pareto (budget = {largest}·d)")
    fig.tight_layout()
    fig.savefig(figs / "05_pareto.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    md.append("\n![Pareto front](figs/05_pareto.png)\n")

    # ─── 6. Stratified ranks ──────────────────────────────────────────────
    md.append(md_section("6. Stratified ranks", level=2))
    md.append(dedent("""
    Mean rank per method, stratified by case attribute.  This reveals
    whether a single overall winner is genuinely best across all
    physics regimes, or only on a subset.

    """))
    strata_dfs = {}
    for label, key in [("VS scenario", "vs_scenario"),
                       ("Topology", "topology"),
                       ("Domain", "domain"),
                       ("Dimension bucket", "dim_bucket")]:
        s = stratified_ranks(df, stratify_by=key, budget_value=largest)
        strata_dfs[label] = s
        if not s.empty:
            md.append(f"\n**{label}** — top-3 methods per stratum:\n\n")
            cols_meta = ["stratum", "n_cases", "friedman_chi2",
                         "friedman_p", "critical_difference"]
            method_cols = [c for c in s.columns if c not in cols_meta]
            rows = []
            for _, row in s.iterrows():
                method_ranks = {m: row[m] for m in method_cols if not pd.isna(row[m])}
                top = sorted(method_ranks.items(), key=lambda kv: kv[1])[:3]
                rows.append({
                    "stratum": row["stratum"],
                    "N": row["n_cases"],
                    "friedman_p": row["friedman_p"],
                    "CD": row["critical_difference"],
                    "best": top[0][0] if top else "—",
                    "2nd": top[1][0] if len(top) > 1 else "—",
                    "3rd": top[2][0] if len(top) > 2 else "—",
                })
            md.append(md_table(pd.DataFrame(rows), max_rows=20))

    try:
        fig = plot_multi_stratum_grid(strata_dfs,
                                      save_path=figs / "06_strat_grid.png")
        plt.close(fig)
        md.append("\n![Stratum grid](figs/06_strat_grid.png)\n")
    except Exception as exc:
        md.append(f"\n_(stratum grid plot failed: {exc})_\n")

    # ─── 7. Per-output rank shifts ────────────────────────────────────────
    md.append(md_section("7. Per-output disaggregation", level=2))
    md.append(dedent("""
    Each case has 3 outputs.  The headline NRMSE averages over them,
    which can hide methods that are great on one output and terrible
    on another.  Here we report:

    - *mean_worst*: mean of the worst per-case NRMSE across outputs
    - *mean_spread*: mean of the (max - min) NRMSE across outputs
    - *rank_shift*: how much rank degrades on the worst output

    """))
    pos = per_output_summary(df, budget_value=largest)
    md.append("\n**Per-output mean NRMSE (sorted by worst-output performance):**\n\n")
    md.append(md_table(pos))

    rs = per_output_rank_shifts(df, budget_value=largest)
    md.append("\n**Mean rank on each output and rank shift (max-min):**\n\n")
    md.append(md_table(rs))

    # ─── 7b. Learning curves ─────────────────────────────────────────────
    if df["budget_mult"].nunique() >= 2:
        md.append(md_section("7b. Learning curves & convergence rates",
                             level=2))
        md.append(dedent("""
        Power-law convergence: NRMSE ≈ A · n^{-α}.  A method with
        higher α improves faster as the training budget grows.
        Methods whose ranking changes between budgets are
        budget-dependent (no free lunch).

        """))

        # Always compute learning-curve table for the plot
        lc = learning_curve_table(df)

        # Ranking at each budget
        rk = ranking_at_each_budget(df)
        md.append("\n**Mean rank at each budget:**\n\n")
        md.append(md_table(rk.reset_index()))

        ss = rank_stability_score(rk)
        md.append("\n**Rank stability (range across budgets):**\n\n")
        md.append(md_table(ss.reset_index().rename(columns={"index": "method"}),
                           max_rows=n_methods))

        # Power-law fits (only meaningful with ≥3 budgets, but report
        # 2-point slope when only 2 budgets are available)
        fits = fit_power_law(lc)
        if not fits.empty:
            agg = fits.groupby("method").agg(
                alpha_mean=("alpha", "mean"),
                alpha_median=("alpha", "median"),
                alpha_min=("alpha", "min"),
                alpha_max=("alpha", "max"),
                R2_median=("R2", "median"),
                n_cases=("alpha", "count"),
            ).reset_index().sort_values("alpha_mean", ascending=False)
            n_budgets_avail = df["budget_mult"].nunique()
            note = ("" if n_budgets_avail >= 3
                    else " _(only 2 budgets — slope is a 2-point estimate)_")
            md.append(f"\n**Power-law convergence rates (NRMSE ∝ n^{{-α}})**{note}**:**\n\n")
            md.append(md_table(agg, max_rows=n_methods))

        # Plot: learning curves on the most informative case
        try:
            n_budgets_avail = df["budget_mult"].nunique()
            cases_with_pts = (df.groupby("case_id")["budget_mult"].nunique()
                               .pipe(lambda s: s[s >= n_budgets_avail])
                               .index.tolist())
            if cases_with_pts:
                pick = cases_with_pts[0]
                fig, ax = plt.subplots(figsize=(9, 5))
                lc_p = lc[lc["case_id"] == pick]
                for m in sorted(lc_p["method"].unique()):
                    mp = lc_p[lc_p["method"] == m].sort_values("budget_mult")
                    if len(mp) >= 2:
                        ax.plot(mp["n_train"], mp["nrmse_mean"],
                                marker="o", label=m, linewidth=1.0,
                                markersize=4, alpha=0.85)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlabel("n_train (log)")
                ax.set_ylabel("NRMSE (log)")
                ax.set_title(f"Learning curves on `{pick}`  "
                             f"(slope = -α in NRMSE ∝ n^{{-α}})")
                ax.legend(fontsize=7, loc="best", ncol=2)
                ax.grid(alpha=0.3)
                fig.tight_layout()
                fig.savefig(figs / "07b_learning_curves.png",
                            dpi=300, bbox_inches="tight")
                plt.close(fig)
                md.append("\n![Learning curves](figs/07b_learning_curves.png)\n")
        except Exception as exc:
            md.append(f"\n_(learning-curve plot failed: {exc})_\n")

    # ─── 7c. Dimensional scaling ─────────────────────────────────────────
    if df["dim"].nunique() >= 3:
        md.append(md_section("7c. Dimensional scaling", level=2))
        md.append(dedent("""
        Does method ranking depend on case dimension?  We split cases
        into 3 dimension bands (balanced by case count) and report
        the mean rank in each band, plus the rank `range` across bands
        — large `range` ⇒ ranking flips with dimension.

        The NRMSE-vs-dim slope (in log space) measures
        *curse-of-dimensionality severity* per method at the largest
        budget: large positive slope ⇒ accuracy degrades quickly with d.

        """))

        # 1) Mean rank by dimension band
        bands = default_dim_bands(df)
        rk_dim = rank_by_dimension_band(df, bands=bands, budget_value=largest)
        if not rk_dim.empty:
            n_per_band = {name: int(((df.drop_duplicates("case_id")["dim"] >= lo)
                                     & (df.drop_duplicates("case_id")["dim"] <= hi)).sum())
                          for name, (lo, hi) in bands.items()}
            md.append(f"\n**Mean rank by dimension band** "
                      f"(cases per band: "
                      f"{', '.join(f'{n} in {name}' for name, n in n_per_band.items())}"
                      f")\n\n")
            rk_dim_disp = rk_dim.reset_index()
            md.append(md_table(rk_dim_disp, max_rows=n_methods))

        # 2) NRMSE-vs-dim slope (curse-of-dim severity) at largest budget
        nv = nrmse_vs_dim(df, budget_value=largest)
        if not nv.empty:
            md.append(f"\n**Curse-of-dim severity at {largest}·d budget** "
                      f"(log-NRMSE vs dim regression):\n\n")
            md.append(md_table(nv, max_rows=n_methods))

        # 3) Power-law α correlated with dim
        if df["budget_mult"].nunique() >= 2:
            av = alpha_vs_dim(df)
            if not av.empty:
                md.append("\n**Convergence-rate α correlated with dimension** "
                          "(per-method Spearman ρ of α vs case dim):\n\n")
                md.append(md_table(av, max_rows=n_methods))

        # 4) Plot: ranking flip across dimension bands
        try:
            if not rk_dim.empty and len(rk_dim.columns) >= 3:
                fig, ax = plt.subplots(figsize=(9, 5))
                band_cols = [c for c in rk_dim.columns if c != "range"]
                xs = list(range(len(band_cols)))
                rng_series = rk_dim["range"]
                # Highlight only the top-5 most-shifting methods so the
                # plot stays readable.
                k = min(5, len(rk_dim))
                top_movers = rng_series.nlargest(k).index.tolist()
                cmap = plt.cm.tab10
                for m in rk_dim.index:
                    if m in top_movers:
                        color = cmap(top_movers.index(m) % 10)
                        ax.plot(xs, [rk_dim.loc[m, c] for c in band_cols],
                                "-o", color=color, lw=1.8, markersize=5,
                                alpha=0.95, label=f"{m} (Δ={rng_series[m]:.1f})")
                    else:
                        ax.plot(xs, [rk_dim.loc[m, c] for c in band_cols],
                                "-", color="lightgrey", lw=0.6,
                                alpha=0.5, zorder=1)
                ax.invert_yaxis()
                ax.set_xticks(xs)
                ax.set_xticklabels(band_cols)
                ax.set_ylabel("Mean rank (lower = better)")
                ax.set_title(f"Top-{k} ranking flips across dimension bands  "
                             f"(NRMSE @ {largest}·d)")
                ax.grid(alpha=0.3, axis="y")
                ax.legend(loc="best", fontsize=8, ncol=1)
                fig.tight_layout()
                fig.savefig(figs / "07c_dim_scaling.png",
                            dpi=300, bbox_inches="tight")
                plt.close(fig)
                md.append("\n![Rank by dimension band](figs/07c_dim_scaling.png)\n")
        except Exception as exc:
            md.append(f"\n_(dim-scaling plot failed: {exc})_\n")

    # ─── 8. Method robustness (per-rep variance) ─────────────────────────
    md.append(md_section("8. Method robustness (per-rep variance)", level=2))
    md.append(dedent("""
    For each method we compute the std of NRMSE across the repeats
    on each case, then average.  Methods with low headline NRMSE but
    high CV are *brittle to the seed* and should be reported with
    caution.

    """))
    grp = (df[df["budget_mult"] == largest]
           .groupby(["method", "case_id"])["nrmse"]
           .agg(nrmse_mean="mean",
                nrmse_std="std",
                nrmse_cv=lambda x: float(x.std() / x.mean())
                                    if x.mean() > 0 else float("nan"))
           .reset_index())
    method_robust = grp.groupby("method").agg(
        mean_nrmse=("nrmse_mean", "mean"),
        mean_per_rep_std=("nrmse_std", "mean"),
        max_per_rep_std=("nrmse_std", "max"),
        mean_cv=("nrmse_cv", "mean"),
    ).reset_index().sort_values("mean_cv")
    md.append(md_table(method_robust, max_rows=n_methods))

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(method_robust["mean_cv"], method_robust["mean_nrmse"],
               s=80, color="#3b7bba", edgecolor="black", linewidth=0.5)
    for _, row in method_robust.iterrows():
        ax.annotate(row["method"],
                    (row["mean_cv"], row["mean_nrmse"]),
                    fontsize=8, xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("Mean NRMSE coefficient of variation (across reps)")
    ax.set_ylabel("Mean NRMSE")
    ax.set_title("Accuracy vs reproducibility")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figs / "08_robustness.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    md.append("\n![Robustness](figs/08_robustness.png)\n")

    # ─── 9. Cross-metric robustness ───────────────────────────────────────
    md.append(md_section("9. Cross-metric ranking robustness", level=2))
    md.append(dedent("""
    Spearman ρ between rankings induced by alternative metrics.
    Values close to 1 mean the headline conclusion is robust to the
    choice of accuracy metric.

    """))
    candidate_metrics = [
        ("nrmse", "lower"),
        ("rmse", "lower"),
        ("mae", "lower"),
        ("maxae", "lower"),
        ("q95ae", "lower"),
        ("r2", "higher"),
    ]
    df_largest = df[df["budget_mult"] == largest]
    rank_dict = {}
    for metric, direction in candidate_metrics:
        if metric not in df_largest.columns:
            continue
        piv = pivot_metric(df_largest, metric)
        rank_dict[metric] = rank_methods(piv,
                                         lower_is_better=(direction == "lower"))
    from scipy.stats import spearmanr
    metrics = list(rank_dict.keys())
    rho_mat = pd.DataFrame(index=metrics, columns=metrics, dtype=float)
    for m1 in metrics:
        for m2 in metrics:
            rho, _ = spearmanr(rank_dict[m1], rank_dict[m2])
            rho_mat.loc[m1, m2] = float(rho)
    md.append(md_table(rho_mat.reset_index().rename(columns={"index": "metric"})))
    md.append(f"\n**Min Spearman ρ across all metric pairs**: "
              f"{rho_mat.values[~np.eye(len(metrics), dtype=bool)].min():.4f}\n")

    # ─── 10. Fit-quality flags ────────────────────────────────────────────
    md.append(md_section("10. Fit-quality flags", level=2))
    md.append(dedent("""
    Per-(case, method) flag of degenerate / failed fits:

    - `crashed`     : run raised an exception
    - `nrmse>5`     : worse than naive mean predictor
    - `r2<0`        : negative R² (worse than mean)
    - `nan_pred`    : surrogate produced NaN predictions

    """))
    flags = fit_quality_flags(df[df["budget_mult"] == largest])
    if not flags.empty:
        md.append(md_table(flags.head(20)))
    else:
        md.append("\n_no degenerate fits detected at the largest budget_\n")

    fail_rates = failure_rate_per_method(df)
    md.append("\n**Failure rate per method (over all budgets):**\n\n")
    md.append(md_table(fail_rates, max_rows=n_methods))

    # ─── 10b. Sensitivity recovery (if available) ───────────────────────
    _, sens_csv = sensitivity_paths(results_dir)
    if sens_csv.exists():
        md.append(md_section("10b. Sensitivity recovery (Sobol-ST)",
                             level=2))
        md.append(dedent(f"""
        Each surrogate is refit on a 20·d design and total-order
        Sobol indices are computed on its predictions, then compared
        with the ground-truth ``sobol_ST`` from `doe_testbed`
        metadata.  This tests whether the surrogate captures *which*
        variables matter — a fundamentally different criterion from
        prediction error.

        Loaded from `{sens_csv.relative_to(results_dir)}`.

        """))
        sens_df = pd.read_csv(sens_csv)
        agg_dict = {
            "n":              ("kendall_tau",  "count"),
            "tau_mean":       ("kendall_tau",  "mean"),
            "tau_median":     ("kendall_tau",  "median"),
            "top1_rate":      ("top1_correct", "mean"),
            "top3_jaccard":   ("top3_jaccard", "mean"),
            "L1_median":      ("L1",           "median"),
        }
        if "is_flat" in sens_df.columns:
            agg_dict["flat_rate"] = ("is_flat", "mean")
        agg = (sens_df.groupby("method")
                       .agg(**agg_dict)
                       .sort_values("tau_mean", ascending=False)
                       .reset_index())
        md.append(md_table(agg, max_rows=n_methods))

        n_negtau = (sens_df["kendall_tau"] < 0).sum()
        md.append(f"\n**Negative-τ failures**: "
                  f"{n_negtau} / {len(sens_df)} (case, method) pairs "
                  f"have ranking *anti-correlated* with truth — "
                  f"signalling the surrogate captured the wrong "
                  f"importance order.\n")

        # ─── 10c. Multi-criteria Pareto (NRMSE × τ) ─────────────────────
        try:
            from benchmark.multicriteria import (
                accuracy_vs_sensitivity_pareto,
                compose_method_scorecard,
                stability_vs_quality_pareto,
            )
            md.append(md_section("10c. Multi-criteria Pareto: "
                                 "accuracy × importance recovery",
                                 level=2))
            md.append(dedent("""
            Methods are scored on two *qualitatively different*
            criteria simultaneously:

            - mean NRMSE rank (point accuracy; lower = better)
            - mean Kendall τ vs ground-truth Sobol-ST
              (importance recovery; higher = better)

            A method on the Pareto front cannot be improved on one
            criterion without sacrificing the other.  This panel
            highlights the *tradeoff* between accuracy and
            interpretability.

            """))
            ap = accuracy_vs_sensitivity_pareto(ranks, agg)
            if not ap.empty:
                # Pareto-on-top, then dominated rows
                ap_disp = ap.copy()
                ap_disp["pareto"] = ap_disp["on_pareto"].map(
                    {True: "✓", False: ""})
                disp_cols = [c for c in
                             ["method", "nrmse_rank", "tau_mean",
                              "pareto", "dominated_by"]
                             if c in ap_disp.columns]
                md.append(md_table(ap_disp[disp_cols], max_rows=n_methods))

                pareto_methods = ap[ap["on_pareto"]]["method"].tolist()
                md.append(f"\n**Pareto-optimal**: "
                          f"{', '.join(f'`{m}`' for m in pareto_methods)}.\n")
                md.append("Picking only by NRMSE ignores `lhs_opt+gb`-class "
                          "methods that are mediocre on point accuracy but "
                          "*best* at recovering variable importance.\n")

                # Plot the 2D Pareto front
                try:
                    fig, ax = plt.subplots(figsize=(7.5, 5))
                    pareto_mask = ap["on_pareto"].to_numpy()
                    xs = ap["nrmse_rank"].to_numpy()
                    ys = ap["tau_mean"].to_numpy()
                    ax.scatter(xs[~pareto_mask], ys[~pareto_mask],
                               s=70, c="lightgrey", edgecolor="grey",
                               label="dominated", alpha=0.85, zorder=2)
                    ax.scatter(xs[pareto_mask], ys[pareto_mask],
                               s=120, c="tab:red", edgecolor="black",
                               label="Pareto-optimal", zorder=3)
                    # Connect Pareto front
                    pareto_pts = ap[ap["on_pareto"]].sort_values("nrmse_rank")
                    ax.plot(pareto_pts["nrmse_rank"], pareto_pts["tau_mean"],
                            "--", color="tab:red", lw=1.2, alpha=0.6,
                            zorder=2)
                    for _, row in ap.iterrows():
                        ax.annotate(row["method"],
                                    (row["nrmse_rank"], row["tau_mean"]),
                                    fontsize=7, alpha=0.75,
                                    xytext=(4, 4), textcoords="offset points")
                    ax.set_xlabel("Mean NRMSE rank  (lower = better)")
                    ax.set_ylabel("Mean Kendall τ  (higher = better)")
                    ax.set_title(f"Accuracy × Sensitivity-recovery Pareto  "
                                 f"(NRMSE @ {largest}·d, τ vs Sobol-ST)")
                    ax.grid(alpha=0.3)
                    ax.legend(loc="lower left")
                    fig.tight_layout()
                    fig.savefig(figs / "10c_multicriteria_pareto.png",
                                dpi=300, bbox_inches="tight")
                    plt.close(fig)
                    md.append("\n![Multi-criteria Pareto]"
                              "(figs/10c_multicriteria_pareto.png)\n")
                except Exception as exc:
                    md.append(f"\n_(multi-criteria Pareto plot failed: {exc})_\n")

            # Stability × Quality table (only if flat_rate available)
            if "flat_rate" in agg.columns:
                qs = stability_vs_quality_pareto(agg)
                brittle = qs[qs["flat_rate"] >= 0.05]
                if not brittle.empty:
                    md.append("\n**Brittle methods (≥ 5 % surrogate "
                              "collapse rate):**\n\n")
                    md.append(md_table(brittle[["method", "tau_mean",
                                                "flat_rate"]],
                                        max_rows=n_methods))
                    md.append("These should be avoided whenever sensitivity "
                              "studies are downstream of the surrogate.\n")

            # Method scorecard
            try:
                cost = (df[df["budget_mult"] == largest]
                        .groupby("method")["t_total_s"].mean())
                slopes = nrmse_vs_dim(df, budget_value=largest)
                sc = compose_method_scorecard(ranks, agg,
                                                cost_summary=cost,
                                                dim_scaling_slopes=slopes)
                disp = [c for c in sc.columns if not c.endswith("_pct")]
                md.append("\n**Combined method scorecard** "
                          "(sorted by NRMSE rank):\n\n")
                md.append(md_table(sc[disp], max_rows=n_methods))
            except Exception as exc:
                md.append(f"\n_(scorecard failed: {exc})_\n")
        except Exception as exc:
            md.append(f"\n_(multi-criteria section failed: {exc})_\n")

    # ─── 11. Performance heatmap & winners ─────────────────────────────────
    md.append(md_section("11. Per-case performance and winners", level=2))
    fig, ax = plt.subplots(figsize=(0.4 * n_methods + 2, 0.4 * n_cases + 2))
    plot_performance_heatmap(pivot, log_scale=True, ax=ax)
    ax.set_title(f"NRMSE per (case, method) @ {largest}·d (log10)")
    fig.tight_layout()
    fig.savefig(figs / "11_perf_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    md.append("\n![Performance heatmap](figs/11_perf_heatmap.png)\n")

    w = winner_table(df, budget_value=largest)
    md.append("\n**Winner counts:**\n\n")
    counts = w["winner"].value_counts().reset_index()
    counts.columns = ["method", "cases_won"]
    md.append(md_table(counts, max_rows=n_methods))

    fig, ax = plt.subplots(figsize=(8, 0.4 * w["winner"].nunique() + 1))
    plot_winners_per_case(w, ax=ax)
    ax.set_title(f"Cases won per method (NRMSE @ {largest}·d, N={len(w)})")
    fig.tight_layout()
    fig.savefig(figs / "11_winners.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    md.append("\n![Winners](figs/11_winners.png)\n")

    # ─── Footer ───────────────────────────────────────────────────────────
    md.append(md_section("Methodological notes", level=2))
    md.append(dedent("""
    - **Test set**: each case uses 1000 Sobol-sampled test points,
      held fixed across all methods within a case.  See
      ``benchmark.scripts.validate_test_protocol`` for empirical MC
      error analysis.
    - **Reproducibility**: under fixed seed, all methods give
      bit-exact results across reruns (verified in
      ``benchmark.scripts.determinism_audit``).  Variation across
      repeats reflects intentional design seed variation only.
    - **Statistical tests**: the headline Friedman test treats each
      case as one observation; the Nemenyi CD is computed at α=0.05
      using the standard studentized-range approximation.
    - **CIs**: bootstrap CIs resample cases (not repeats) with
      replacement, n_boot=1000.

    Author: Binglin Wang (NUDT).  Auto-generated report.
    """))

    # Save
    report_path = out / "REPORT.md"
    report_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\nReport written to {report_path}")
    print(f"Figures saved under {figs}/")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    generate(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
