"""
benchmark.scripts.validate_test_protocol
==========================================
First-principles audit of the benchmark testing protocol.

Three diagnostic blocks:

A. **Test-set MC sufficiency** — bootstrap-CI of NRMSE estimates as a
   function of ``n_test``.  Goal: establish that the chosen
   ``n_test`` is large enough that test-set noise is small relative
   to between-method differences.

B. **Cross-metric ranking consistency** — does the headline ranking
   change if we swap NRMSE for RMSE / max-AE / q95-AE / R²?  If yes,
   the metric choice itself is a degree of freedom that needs to be
   reported transparently.

C. **Method robustness** — std of NRMSE across the four repeats per
   (case, method, budget).  Methods with low mean rank but high
   per-rep variance are practically risky: the ranking is brittle to
   the seed.

The script runs entirely on saved CSVs from completed benchmark runs;
no model fitting required.
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Sequence

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import doe_testbed as db
from benchmark.analysis import pivot_metric, rank_methods
from benchmark.plots import use_paper_style
from benchmark.rigour import bootstrap_metric_ci
from scripts.prepare_figdata import (
    validate_headline_results_source,
    validate_results_contract,
)


# ─── Block A: test-set MC sufficiency ────────────────────────────────────
def block_a_test_set_sufficiency(out_dir: Path) -> None:
    """For 4 representative cases × 1 strong method (lhs_opt+gp), study
    how NRMSE-CI shrinks as n_test grows from 100 → 5000."""
    print("\n" + "=" * 72)
    print("BLOCK A — test-set MC sufficiency")
    print("=" * 72)

    from benchmark.samplers import OptimisedLHSSampler
    from benchmark.surrogates import GPSurrogate
    representative_cases = [
        "cantilever_beam_5d",        # smooth, low-d
        "fin_heat_sink_4d",          # smooth, multi-output
        "ten_bar_truss_10d",         # higher-d, sparse
        "tuned_mass_damper_3d",      # heterogeneous topology
    ]
    n_test_levels = [100, 200, 500, 1000, 2000, 5000]

    rows = []
    fig, axes = plt.subplots(1, len(representative_cases),
                             figsize=(4 * len(representative_cases), 4),
                             sharey=True)

    for idx, case_id in enumerate(representative_cases):
        ax = axes[idx]
        case = db.get_case(case_id)
        d = case.meta.dim_input
        # Train once at 20d budget (sampler returns raw [0,1]^d samples)
        sampler = OptimisedLHSSampler()
        X_train = sampler.sample(20 * d, d, seed=0)
        Y_train = case.evaluate(X_train)
        gp = GPSurrogate()
        gp.fit(X_train, Y_train)

        # Generate the largest test set once
        X_test_full = case.sample(max(n_test_levels), method="sobol", seed=0)
        Y_test_full = case.evaluate(X_test_full)
        Y_pred_full, _ = gp.predict(X_test_full)

        means, los, his = [], [], []
        for n in n_test_levels:
            stats = bootstrap_metric_ci(
                Y_test_full[:n], Y_pred_full[:n],
                n_boot=1000, seed=0,
            )
            means.append(stats["nrmse_mean"])
            los.append(stats["nrmse_ci_low"])
            his.append(stats["nrmse_ci_high"])
            rows.append({
                "case_id": case_id, "n_test": n,
                "nrmse_mean": stats["nrmse_mean"],
                "nrmse_ci_low": stats["nrmse_ci_low"],
                "nrmse_ci_high": stats["nrmse_ci_high"],
                "nrmse_ci_width": stats["nrmse_ci_width"],
                "nrmse_relative_ci_width": stats["nrmse_relative_ci_width"],
            })

        ax.errorbar(n_test_levels, means,
                    yerr=[np.array(means) - np.array(los),
                          np.array(his) - np.array(means)],
                    fmt="o-", color="#3b7bba", ecolor="0.4",
                    capsize=4, markersize=6, linewidth=1.2,
                    markeredgecolor="black", markeredgewidth=0.4)
        ax.axvline(1000, color="red", ls="--", lw=1, alpha=0.6,
                   label="benchmark default")
        ax.set_xscale("log")
        ax.set_title(case_id, fontsize=10)
        ax.set_xlabel("n_test")
        if idx == 0:
            ax.set_ylabel("NRMSE  (95% bootstrap CI)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("Test-set MC error vs n_test (lhs_opt+gp @ 20d)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "test_set_sufficiency.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "test_set_sufficiency.csv", index=False)
    print(df.round(4).to_string(index=False))

    # Headline conclusion
    df_n1000 = df[df["n_test"] == 1000]
    print("\nAt n_test=1000:")
    print(f"  median relative CI width: {df_n1000['nrmse_relative_ci_width'].median()*100:.1f}%")
    print(f"  max    relative CI width: {df_n1000['nrmse_relative_ci_width'].max()*100:.1f}%")
    print("  → if max < 5%, n_test=1000 is sufficient (between-method gaps usually >5%)")


# ─── Block B: cross-metric ranking consistency ───────────────────────────
def block_b_cross_metric_consistency(df: pd.DataFrame, out_dir: Path) -> None:
    print("\n" + "=" * 72)
    print("BLOCK B — cross-metric ranking consistency")
    print("=" * 72)

    largest = int(max(df["budget_mult"]))
    df_b = df[df["budget_mult"] == largest]
    df_b = df_b[df_b["status"] == "ok"]

    candidate_metrics = [
        ("nrmse", "lower"),
        ("rmse", "lower"),
        ("mae", "lower"),
        ("maxae", "lower"),
        ("q95ae", "lower"),
        ("r2", "higher"),
    ]
    available = [(m, d) for m, d in candidate_metrics if m in df_b.columns]

    # Compute mean rank per (method, metric)
    rank_dict = {}
    for metric, direction in available:
        piv = pivot_metric(df_b, metric)
        ranks = rank_methods(piv, lower_is_better=(direction == "lower"))
        rank_dict[metric] = ranks

    rank_df = pd.DataFrame(rank_dict)
    rank_df.index.name = "method"
    rank_df["span"] = rank_df.max(axis=1) - rank_df.min(axis=1)
    rank_df.reset_index().to_csv(out_dir / "cross_metric_ranks.csv", index=False)
    print("\nMean rank per (method, metric):")
    print(rank_df.round(2).to_string())

    # Rank-correlation matrix
    from scipy.stats import spearmanr
    metrics = list(rank_dict.keys())
    corr = pd.DataFrame(index=metrics, columns=metrics, dtype=float)
    for m1 in metrics:
        for m2 in metrics:
            rho, _ = spearmanr(rank_dict[m1], rank_dict[m2])
            corr.loc[m1, m2] = float(rho)
    print("\nSpearman ρ between metric-induced rankings:")
    print(corr.round(3).to_string())
    corr.index.name = "metric"
    corr.reset_index().to_csv(out_dir / "cross_metric_rank_correlation.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.to_numpy(dtype=float), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_yticklabels(metrics)
    for i in range(len(metrics)):
        for j in range(len(metrics)):
            v = corr.iloc[i, j]
            color = "white" if abs(v) > 0.5 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=color, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Spearman ρ between metric-induced method rankings")
    fig.tight_layout()
    fig.savefig(out_dir / "cross_metric_correlation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Methods whose rank changes the most across metrics
    print("\nMethods with the largest rank span across metrics:")
    print(rank_df.sort_values("span", ascending=False).head(5).round(2).to_string())


# ─── Block C: method robustness (per-rep variance) ───────────────────────
def block_c_method_robustness(df: pd.DataFrame, out_dir: Path) -> None:
    print("\n" + "=" * 72)
    print("BLOCK C — method robustness (NRMSE std across reps)")
    print("=" * 72)

    largest = int(max(df["budget_mult"]))
    df_c = df[(df["budget_mult"] == largest) & (df["status"] == "ok")]

    grp = df_c.groupby(["method", "case_id"])["nrmse"].agg(
        nrmse_mean="mean",
        nrmse_std="std",
        nrmse_cv=lambda x: float(x.std() / x.mean()) if x.mean() > 0 else float("nan"),
    ).reset_index()

    # Aggregate across cases per method
    method_robust = grp.groupby("method").agg(
        mean_nrmse=("nrmse_mean", "mean"),
        mean_per_rep_std=("nrmse_std", "mean"),
        median_per_rep_std=("nrmse_std", "median"),
        max_per_rep_std=("nrmse_std", "max"),
        mean_cv=("nrmse_cv", "mean"),
    ).reset_index().sort_values("mean_cv")

    print("Per-method robustness (mean over cases):")
    print(method_robust.round(4).to_string(index=False))
    method_robust.to_csv(out_dir / "method_robustness.csv", index=False)

    # Plot mean NRMSE vs CV
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(method_robust["mean_cv"], method_robust["mean_nrmse"],
               s=80, color="#3b7bba",
               edgecolor="black", linewidth=0.5)
    for _, row in method_robust.iterrows():
        ax.annotate(row["method"],
                    (row["mean_cv"], row["mean_nrmse"]),
                    fontsize=8, xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("Mean NRMSE coefficient of variation (across reps)")
    ax.set_ylabel("Mean NRMSE")
    ax.set_title("Method robustness — accuracy vs reproducibility")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "method_robustness.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _replace_analysis_dir(tmp_dir: Path, out_dir: Path) -> None:
    """Replace ``out_dir`` with ``tmp_dir`` while preserving old outputs on failure."""
    backup_dir = out_dir.parent / f".{out_dir.name}.bak"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    try:
        if out_dir.exists():
            shutil.move(str(out_dir), str(backup_dir))
        shutil.move(str(tmp_dir), str(out_dir))
    except Exception:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        if backup_dir.exists():
            shutil.move(str(backup_dir), str(out_dir))
        raise
    finally:
        if backup_dir.exists():
            shutil.rmtree(backup_dir)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Validate test-set MC noise, cross-metric ranking consistency, "
            "and method robustness for a completed benchmark run."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing results.csv, usually benchmark/results/full_benchmark.",
    )
    p.add_argument(
        "--require-headline-source",
        action="store_true",
        help=(
            "Fail unless <results_dir>/results.csv is the canonical "
            "full_benchmark/results.csv and the CSV is the exact frozen "
            "34 x 20 x 4 x 30 nature-pool grid. Use this in the formal "
            "paper pipeline."
        ),
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    src = Path(args.results_dir)
    results_csv = src / "results.csv"
    if not results_csv.exists():
        print(f"[test_protocol:error] results.csv does not exist: {results_csv}", file=sys.stderr)
        return 2
    df = pd.read_csv(src / "results.csv")
    try:
        validate_results_contract(df)
        if args.require_headline_source:
            validate_headline_results_source(results_csv, df)
    except AssertionError as exc:
        print(f"[test_protocol:error] results.csv contract failed: {exc}", file=sys.stderr)
        return 2
    out = src / "analysis"
    use_paper_style()

    tmp_parent = Path(tempfile.mkdtemp(prefix=".analysis.tmp.", dir=str(src)))
    try:
        block_a_test_set_sufficiency(tmp_parent)
        block_b_cross_metric_consistency(df, tmp_parent)
        block_c_method_robustness(df, tmp_parent)

        _replace_analysis_dir(tmp_parent, out)
        print(f"\nAll protocol-validation outputs saved to {out}/")
    finally:
        if tmp_parent.exists():
            shutil.rmtree(tmp_parent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
