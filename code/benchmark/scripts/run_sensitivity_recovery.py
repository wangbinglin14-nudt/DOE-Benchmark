"""
benchmark.scripts.run_sensitivity_recovery
============================================
For every (case, method) at the largest budget, refit the surrogate
and compute total-order Sobol indices on it.  Compare with the
ground-truth ``sobol_ST`` from ``doe_testbed`` metadata.

Why this matters
----------------
A surrogate may have low *prediction* error on a held-out test set
yet attribute output variance to the wrong inputs.  Engineering
decisions (tolerance allocation, parametric design, what-if
analyses) depend on variable importance, so this is a first-class
quality criterion.

Outputs
-------
- ``sensitivity_recovery.csv``   long-form per-(case, method)
- ``sensitivity_recovery.png``   per-method Kendall-τ box plot
- ``sensitivity_topk_heatmap.png`` per-(case, method) top-1 hits
- ``sensitivity_summary.csv``    per-method aggregate scores

Usage
-----
    python benchmark/scripts/run_sensitivity_recovery.py
    python benchmark/scripts/run_sensitivity_recovery.py \\
        --pool nature --budget-mult 20 \\
        --output-dir benchmark/results/full_benchmark/sensitivity_recovery
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import warnings
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError:
    pass

import matplotlib.pyplot as plt
import numpy as np

import doe_testbed as db
from benchmark.methods import (
    bo_method_pool,
    default_method_pool,
    extended_method_pool,
    full_method_pool,
    nature_pool,
)
from benchmark.plots import use_paper_style
from benchmark.sensitivity_recovery import run_sensitivity_recovery

POOLS = {
    "default": default_method_pool,
    "extended": extended_method_pool,
    "full": full_method_pool,
    "bo": bo_method_pool,
    "nature": nature_pool,
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_sensitivity_recovery",
        description=(
            "Refit each surrogate at one budget and compare its "
            "Sobol' indices against the ground-truth sobol_ST stored "
            "in doe_testbed metadata."
        ),
    )
    p.add_argument(
        "--pool",
        choices=list(POOLS.keys()),
        default="extended",
        help=(
            "Method pool to evaluate.  Default 'extended' "
            "(15 static-design methods, the legacy default for this "
            "script)."
        ),
    )
    p.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Restrict to a subset of case ids; default = all cases.",
    )
    p.add_argument(
        "--budget-mult",
        type=int,
        default=20,
        help="Training budget as a multiplier of input dimension d.",
    )
    p.add_argument(
        "--n-base",
        type=int,
        default=256,
        help="Sobol' base sample size for the indices computation.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for reproducibility.",
    )
    p.add_argument(
        "--output-idx",
        type=int,
        default=0,
        help="Which output (0-based) to compute Sobol indices on.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "benchmark" / "results" / "sensitivity_recovery",
        help="Directory to write the four CSV/PNG artefacts to.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None):
    args = _parse_args(argv)
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cases:
        case_ids = list(args.cases)
    else:
        case_ids = sorted(c.case_id for c in db.list_cases())
    methods = POOLS[args.pool]()

    print("=" * 72)
    print("Sensitivity-Recovery Sweep")
    print("=" * 72)
    print(f"Pool    : {args.pool}")
    print(f"Cases   : {len(case_ids)}")
    print(f"Methods : {len(methods)}")
    print(f"Budget  : {args.budget_mult}*d   "
          f"n_base={args.n_base}  seed={args.seed}")
    print(f"Total   : {len(case_ids) * len(methods)} (case, method) refits")
    print(f"Output  : {out_dir}")
    print("=" * 72)

    t0 = time.perf_counter()
    df = run_sensitivity_recovery(
        case_ids=case_ids,
        methods=methods,
        budget_mult=args.budget_mult,
        n_base=args.n_base,
        seed=args.seed,
        output_idx=args.output_idx,
        verbose=True,
    )
    print(f"\n=== Sensitivity recovery completed in {(time.perf_counter()-t0)/60:.1f} min ===")
    df.to_csv(out_dir / "sensitivity_recovery.csv", index=False)
    print(f"Saved long-form: {out_dir/'sensitivity_recovery.csv'}  ({len(df)} rows)")

    # ── Aggregate per-method summary ─────────────────────────────────
    print("\n=== Per-method aggregate ===")
    agg_kwargs = {
        "n": ("kendall_tau", "count"),
        "tau_mean": ("kendall_tau", "mean"),
        "tau_median": ("kendall_tau", "median"),
        "top1_rate": ("top1_correct", "mean"),
        "top3_jaccard_mean": ("top3_jaccard", "mean"),
        "L1_mean": ("L1", "mean"),
        "L1_median": ("L1", "median"),
        "L2_mean": ("L2", "mean"),
    }
    # negative-τ rate (anti-correlated rankings)
    agg_kwargs["negative_tau_rate"] = (
        "kendall_tau",
        lambda s: float((s < 0).mean()),
    )
    if "is_flat" in df.columns:
        agg_kwargs["flat_rate"] = ("is_flat", "mean")
    summary = (df.groupby("method")
                 .agg(**agg_kwargs)
                 .sort_values("tau_mean", ascending=False)
                 .reset_index())
    print(summary.round(3).to_string(index=False))
    summary.to_csv(out_dir / "sensitivity_summary.csv", index=False)

    use_paper_style()

    # ── Box plot of Kendall tau per method ───────────────────────────
    methods_sorted = summary["method"].tolist()
    fig, ax = plt.subplots(figsize=(11, 5))
    bp_data = [df[df["method"] == m]["kendall_tau"].dropna().to_numpy()
               for m in methods_sorted]
    ax.boxplot(bp_data, labels=methods_sorted, showmeans=True,
               meanprops={"marker": "D", "markerfacecolor": "red",
                          "markersize": 4, "markeredgecolor": "black"})
    ax.axhline(0, color="grey", lw=0.6, ls="--")
    ax.axhline(1, color="green", lw=0.4, ls=":", alpha=0.5)
    ax.set_xticklabels(methods_sorted, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Kendall τ (predicted-ST vs true-ST)")
    ax.set_title(f"Sensitivity-recovery Kendall τ by method  "
                 f"(per case, N={len(case_ids)}; 20·d budget)")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "sensitivity_recovery.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ── Top-1 heatmap (case × method) ────────────────────────────────
    pivot_top1 = df.pivot_table(
        index="case_id", columns="method", values="top1_correct",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(0.5 * pivot_top1.shape[1] + 2,
                                     0.32 * pivot_top1.shape[0] + 1.5))
    im = ax.imshow(pivot_top1.to_numpy(dtype=float), aspect="auto",
                   cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(np.arange(pivot_top1.shape[1]))
    ax.set_xticklabels(pivot_top1.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(pivot_top1.shape[0]))
    ax.set_yticklabels(pivot_top1.index, fontsize=7)
    cbar = plt.colorbar(im, ax=ax, fraction=0.04)
    cbar.set_label("Top-1 variable correct (1.0 = yes)")
    ax.set_title("Did the surrogate identify the most important variable?")
    fig.tight_layout()
    fig.savefig(out_dir / "sensitivity_topk_heatmap.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ── Anomaly detection: high NRMSE-rank but low τ-rank ────────────
    # Reports method × case combinations where prediction error is low
    # but sensitivity recovery is poor — the most dangerous case.
    print("\n=== Worst sensitivity-recovery failures (Kendall τ < 0) ===")
    fails = df[df["kendall_tau"] < 0].sort_values("kendall_tau")
    if not fails.empty:
        cols = ["case_id", "method", "kendall_tau", "L1", "argmax_true", "argmax_pred"]
        print(fails[cols].to_string(index=False))
        fails.to_csv(out_dir / "sensitivity_failures.csv", index=False)
    else:
        print("  (no negative-τ cases found)")

    print(f"\nAll files saved under {out_dir}/")


if __name__ == "__main__":
    main()
