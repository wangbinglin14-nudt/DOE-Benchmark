"""
benchmark.plots
===============
Publication-quality plots for benchmark results.

All plots use a common style suitable for high-impact journals:
- Inter / Helvetica-style fonts via matplotlib defaults
- Restrained palette (Tableau colorblind-safe)
- 300 DPI export, sans-serif, no gridlines unless informative
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark.analysis import cd_diagram_data


def plot_rank_ci(
    rank_ci_df: pd.DataFrame,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
):
    """Forest plot of mean ranks with bootstrap 95% CI.

    Expects a DataFrame with columns ``method``, ``mean_rank``,
    ``ci_low``, ``ci_high`` (as produced by
    :func:`benchmark.stratified.bootstrap_rank_ci`).
    """
    df = rank_ci_df.dropna(subset=["mean_rank"]).copy()
    df = df.sort_values("mean_rank")
    n = len(df)
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 0.35 * n + 1.2))
    y = np.arange(n)
    err_low = (df["mean_rank"] - df["ci_low"]).to_numpy()
    err_high = (df["ci_high"] - df["mean_rank"]).to_numpy()
    ax.errorbar(
        df["mean_rank"], y,
        xerr=[err_low, err_high],
        fmt="o", color="#3b7bba",
        ecolor="0.5", elinewidth=1.5, capsize=3,
        markersize=7, markeredgecolor="black", markeredgewidth=0.5,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(df["method"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean rank (95% bootstrap CI)")
    ax.grid(axis="x", alpha=0.3)
    if title:
        ax.set_title(title)
    return ax


def plot_winners_per_case(
    df_winners: pd.DataFrame,
    methods_order: Optional[list] = None,
    ax: Optional[plt.Axes] = None,
):
    """Horizontal bar plot of cases-won per method, ranked.

    Expects a DataFrame with column ``winner``.
    """
    counts = df_winners["winner"].value_counts()
    if methods_order:
        counts = counts.reindex(methods_order, fill_value=0)
    counts = counts[counts > 0].sort_values(ascending=True)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 0.35 * len(counts) + 1))
    ax.barh(counts.index, counts.values, color="#3b7bba",
            edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Cases won (lowest NRMSE)")
    for y, v in enumerate(counts.values):
        ax.text(v + 0.05, y, str(int(v)), va="center", fontsize=9)
    return ax


def plot_pairwise_win_matrix(
    win: pd.DataFrame,
    order_by: str = "dominance",
    ax: Optional[plt.Axes] = None,
):
    """Heatmap of pairwise win-rates.

    Diagonal = 0.5; off-diagonal entry M[i,j] = fraction of cases on
    which method i beats method j.
    """
    if order_by == "dominance":
        # Sort rows/cols by mean win-rate (excluding self)
        n = len(win)
        if n > 1:
            score = (win.sum(axis=1) - 0.5) / (n - 1)
            order = score.sort_values(ascending=False).index
            win = win.loc[order, order]

    if ax is None:
        fig, ax = plt.subplots(figsize=(0.4 * len(win) + 2, 0.4 * len(win) + 2))
    arr = win.to_numpy(dtype=float)
    # Center colour scale at 0.5
    im = ax.imshow(arr, cmap="RdBu_r", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(win)))
    ax.set_yticks(np.arange(len(win)))
    ax.set_xticklabels(win.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(win.index, fontsize=8)
    # Annotate cells
    for i in range(len(win)):
        for j in range(len(win)):
            v = arr[i, j]
            color = "white" if (v > 0.7 or v < 0.3) else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color=color)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("P(row beats column)")
    ax.set_title("Pairwise win-rate matrix")
    return ax


def plot_pareto(
    pareto_df: pd.DataFrame,
    metric_col: str = "mean_metric",
    cost_col: str = "mean_cost",
    log_x: bool = True,
    log_y: bool = False,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
):
    """Compute-vs-accuracy scatter with Pareto front highlighted.

    Expects a DataFrame from :func:`benchmark.rigour.pareto_front`
    with at least the columns ``method``, ``metric_col``, ``cost_col``,
    ``on_pareto``.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # All points
    other = pareto_df[~pareto_df["on_pareto"]]
    front = pareto_df[pareto_df["on_pareto"]].sort_values(cost_col)

    ax.scatter(other[cost_col], other[metric_col],
               s=70, c="#999999", marker="o",
               edgecolor="black", linewidth=0.5, label="dominated")
    ax.scatter(front[cost_col], front[metric_col],
               s=110, c="#d62728", marker="*",
               edgecolor="black", linewidth=0.6, label="Pareto front", zorder=5)

    # Connect Pareto front
    if len(front) >= 2:
        ax.plot(front[cost_col], front[metric_col],
                color="#d62728", linewidth=1.2, alpha=0.7, zorder=4)

    # Annotate methods
    for _, row in pareto_df.iterrows():
        ax.annotate(row["method"], (row[cost_col], row[metric_col]),
                    fontsize=8, xytext=(5, 4), textcoords="offset points")

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Mean compute (s per run)")
    ax.set_ylabel("Mean NRMSE")
    ax.legend(frameon=False, loc="best")
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(title)
    return ax


def plot_multi_stratum_grid(
    strata_dfs: Dict[str, pd.DataFrame],
    save_path: Optional[Path] = None,
):
    """Grid of stratified rank heatmaps, one panel per stratification dimension.

    Parameters
    ----------
    strata_dfs : dict
        Map ``label -> stratified_ranks DataFrame``.  Typical labels:
        ``'VS scenario'``, ``'Topology'``, ``'Domain'``, ``'Dim bucket'``.
    """
    n_panels = len(strata_dfs)
    fig, axes = plt.subplots(n_panels, 1, figsize=(11, 2.5 + 1.5 * n_panels))
    if n_panels == 1:
        axes = [axes]

    for ax, (label, s) in zip(axes, strata_dfs.items(), strict=False):
        if s.empty:
            ax.text(0.5, 0.5, f"{label}: no data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        cols_meta = ["stratum", "n_cases", "friedman_chi2", "friedman_p", "critical_difference"]
        method_cols = [c for c in s.columns if c not in cols_meta]
        arr = s[method_cols].to_numpy(dtype=float)
        im = ax.imshow(arr, aspect="auto", cmap="RdYlGn_r")
        ax.set_xticks(np.arange(len(method_cols)))
        ax.set_xticklabels(method_cols, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(s)))
        ax.set_yticklabels(
            [f"{r['stratum']} (N={r['n_cases']})" for _, r in s.iterrows()],
            fontsize=8,
        )
        cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label("Mean rank", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        ax.set_title(label, fontsize=10, loc="left")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# ─── Style ───────────────────────────────────────────────────────────────
def use_paper_style() -> None:
    """Apply matplotlib rcParams suitable for paper figures."""
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })


# ─── Critical-difference diagram (Demšar 2006) ───────────────────────────
def plot_cd_diagram(
    pivot: pd.DataFrame,
    alpha: float = 0.05,
    lower_is_better: bool = True,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
):
    """Critical-difference diagram for k methods over N cases.

    Methods connected by horizontal bars are not significantly
    different at level ``alpha``.
    """
    data = cd_diagram_data(pivot, alpha=alpha, lower_is_better=lower_is_better)
    ranks = data["ranks"].dropna()
    if ranks.empty:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 1))
        ax.text(0.5, 0.5, "Insufficient data for CD diagram",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return ax
    cd = data["cd"]
    n_methods = len(ranks)
    methods = ranks.index.tolist()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 0.6 + 0.3 * n_methods))

    # X-axis: ranks (lower = better, on the left)
    rmin = float(np.floor(ranks.min() - 0.5))
    rmax = float(np.ceil(ranks.max() + 0.5))
    ax.set_xlim(rmin, rmax)
    ax.invert_xaxis()  # better methods on the left
    ax.set_ylim(-0.5, n_methods + 1)

    # Top axis ticks for ranks
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xticks(np.arange(rmin, rmax + 0.5, 1.0))
    ax.set_xlabel(f"Mean rank (CD={cd:.2f}, α={alpha}, N={data['n_cases']})")

    # Top horizontal axis line
    ax.axhline(n_methods + 0.7, color="black", lw=1.0, xmin=0, xmax=1)

    # Place method labels: half on the left, half on the right
    half = (n_methods + 1) // 2
    for i, m in enumerate(methods):
        rank = ranks[m]
        if i < half:
            # left side
            y = n_methods - i
            ax.plot([rank, rmax], [n_methods + 0.7, y], color="0.4", lw=0.5)
            ax.plot([rmax, rmax + 0.1], [y, y], color="0.4", lw=0.5)
            ax.text(rmax + 0.15, y, f"{m}  ({rank:.2f})",
                    va="center", ha="left", fontsize=9)
        else:
            # right side
            y = i - half + 1
            ax.plot([rank, rmin], [n_methods + 0.7, y], color="0.4", lw=0.5)
            ax.plot([rmin, rmin - 0.1], [y, y], color="0.4", lw=0.5)
            ax.text(rmin - 0.15, y, f"({rank:.2f})  {m}",
                    va="center", ha="right", fontsize=9)

    # CD bars below the rank axis
    bar_y_start = n_methods + 0.4
    bar_y_step = -0.18
    drawn = 0
    for clique in data["cliques"]:
        if len(clique) <= 1:
            continue
        rs = [ranks[m] for m in clique]
        x0, x1 = min(rs) - 0.05, max(rs) + 0.05
        y = bar_y_start + drawn * bar_y_step
        ax.plot([x0, x1], [y, y], color="black", lw=2.5, solid_capstyle="butt")
        drawn += 1

    if title:
        ax.set_title(title, pad=20)
    return ax


# ─── Win-rate bar plot ───────────────────────────────────────────────────
def plot_win_count(summary: Dict, ax: Optional[plt.Axes] = None):
    """Bar chart of how many cases each method wins."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 0.3 * len(summary["win_count"]) + 1.0))
    counts = summary["win_count"].sort_values(ascending=True)
    ax.barh(counts.index, counts.values, color="#3b7bba", edgecolor="black", lw=0.5)
    ax.set_xlabel("Number of cases won (lowest error)")
    ax.set_title(f"Win counts for {summary['metric']} "
                 f"(N={summary['n_cases']} cases, k={summary['n_methods']} methods)")
    return ax


# ─── Performance heatmap ─────────────────────────────────────────────────
def plot_performance_heatmap(
    pivot: pd.DataFrame,
    log_scale: bool = True,
    ax: Optional[plt.Axes] = None,
    cmap: str = "viridis_r",
):
    """Cases × methods heatmap of the metric.  Lower is darker (= better)."""
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(0.45 * pivot.shape[1] + 2, 0.3 * pivot.shape[0] + 1.5)
        )
    arr = pivot.to_numpy(dtype=float)
    if log_scale:
        arr = np.log10(np.maximum(arr, 1e-10))
    im = ax.imshow(arr, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log₁₀(NRMSE)" if log_scale else "NRMSE")
    return ax


# ─── Convenience: full report figure ─────────────────────────────────────
def plot_summary_report(summary: Dict, save_dir: Optional[Path] = None) -> None:
    """Generate the standard 3-panel summary figure for a metric."""
    use_paper_style()
    fig = plt.figure(figsize=(14, 8))

    ax1 = fig.add_subplot(2, 2, 1)
    plot_cd_diagram(summary["pivot"], lower_is_better=True,
                    title=f"Critical-Difference: {summary['metric']}", ax=ax1)

    ax2 = fig.add_subplot(2, 2, 2)
    plot_win_count(summary, ax=ax2)

    ax3 = fig.add_subplot(2, 1, 2)
    plot_performance_heatmap(summary["pivot"], ax=ax3)

    fig.tight_layout()
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / f"summary_{summary['metric']}.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")
    return fig
