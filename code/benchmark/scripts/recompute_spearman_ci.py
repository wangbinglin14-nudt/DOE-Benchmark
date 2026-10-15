"""Add bootstrap 95 % CI on Spearman ρ and Kendall τ to
``correlation.json`` without re-running the ranking-reversal benchmark.

Rationale
---------
The ranking-reversal experiment reports Spearman ρ between the
analytical-rank vector and the physical-rank vector over the resolved
method axis.  The Nature-family headline run uses N = 20 methods from
``nature_pool()``; older diagnostics may contain smaller pools.  A
point-estimate-only report is statistically underpowered because the
sampling distribution of ρ is wide for the method counts used here, and
a reviewer will immediately ask for a CI.

We bootstrap by resampling **methods** (the N-axis) with replacement
``n_boot`` times, recomputing ρ and τ on each resample, and taking the
2.5 / 97.5 percentiles.  This is the direct analogue of asking "if we
had sampled a different set of methods, what range of ρ would we
observe?"

The script reads existing ``rank_comparison.csv``, augments
``correlation.json`` with ``spearman_rho_ci_lo/hi``,
``kendall_tau_ci_lo/hi`` and ``n_methods`` fields, and leaves every
other field untouched so downstream consumers
(``scripts/fill_paper_numbers.py``) see the new keys transparently.

Usage
-----
    python -m benchmark.scripts.recompute_spearman_ci \
        [--rr-dir benchmark/results/ranking_reversal] \
        [--n-boot 10000] [--seed 0]
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


def _effective_case_counts(rr_dir: Path, mode: str, budget: int) -> Dict[str, int]:
    """Count observed and complete-case rows used by the D1 rank estimator.

    ``rank_methods`` applies the registered complete-case policy by dropping
    any case with a missing method metric.  The correlation summary must expose
    that effective case count instead of implying that every nominal pairing
    entered the rank vector.
    """
    paths = {
        "analytical": rr_dir / "analytical" / "results.csv",
        "physical": rr_dir / f"pairing_{mode}" / "physical" / "results.csv",
    }
    out: Dict[str, int] = {}
    for label, path in paths.items():
        if not path.exists():
            out[f"n_{label}_cases_observed"] = 0
            out[f"n_{label}_cases_complete"] = 0
            continue
        frame = pd.read_csv(path, low_memory=False)
        frame = frame[frame["budget_mult"] == budget]
        observed = frame["case_id"].nunique()
        ok = frame[frame["status"] == "ok"] if "status" in frame.columns else frame
        pivot = ok.pivot_table(
            index="case_id", columns="method", values="nrmse", aggfunc="median"
        )
        out[f"n_{label}_cases_observed"] = int(observed)
        out[f"n_{label}_cases_complete"] = int(pivot.dropna(how="any").shape[0])
    return out


def bootstrap_rho_ci(
    rank_a: np.ndarray,
    rank_b: np.ndarray,
    *,
    n_boot: int = 10_000,
    seed: int = 0,
    ci: float = 0.95,
) -> Dict[str, float]:
    """Percentile-bootstrap CI on Spearman ρ / Kendall τ by resampling
    methods (the N-axis) with replacement.

    Returns a dict with point estimates, CI bounds, and diagnostics.
    """
    rank_a = np.asarray(rank_a, dtype=float)
    rank_b = np.asarray(rank_b, dtype=float)
    if rank_a.shape != rank_b.shape or rank_a.ndim != 1:
        raise ValueError("rank_a and rank_b must be 1-D arrays of the same length")
    n = rank_a.size

    rho_point, p_rho = spearmanr(rank_a, rank_b)
    tau_point, p_tau = kendalltau(rank_a, rank_b)

    rng = np.random.default_rng(seed)
    rho_boot = np.empty(n_boot, dtype=float)
    tau_boot = np.empty(n_boot, dtype=float)
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        # Degenerate resample (all same method): Spearman undefined → NaN.
        if len(np.unique(idx)) < 2:
            rho_boot[k] = np.nan
            tau_boot[k] = np.nan
            continue
        rho_boot[k], _ = spearmanr(rank_a[idx], rank_b[idx])
        tau_boot[k], _ = kendalltau(rank_a[idx], rank_b[idx])

    alpha = 1.0 - ci
    lo_q = 100 * alpha / 2
    hi_q = 100 * (1 - alpha / 2)
    rho_lo, rho_hi = np.nanpercentile(rho_boot, [lo_q, hi_q])
    tau_lo, tau_hi = np.nanpercentile(tau_boot, [lo_q, hi_q])

    return {
        "spearman_rho": float(rho_point),
        "spearman_p": float(p_rho),
        "spearman_rho_ci_lo": float(rho_lo),
        "spearman_rho_ci_hi": float(rho_hi),
        "kendall_tau": float(tau_point),
        "kendall_p": float(p_tau),
        "kendall_tau_ci_lo": float(tau_lo),
        "kendall_tau_ci_hi": float(tau_hi),
        "n_methods": int(n),
        "n_boot": int(n_boot),
        "ci_level": float(ci),
        "boot_degenerate_frac": float(np.isnan(rho_boot).mean()),
    }


def recompute(rr_dir: Path, *, n_boot: int, seed: int) -> int:
    rank_cmp = rr_dir / "rank_comparison.csv"
    corr_json = rr_dir / "correlation.json"
    if not corr_json.exists():
        print(f"[err] {corr_json} not found")
        return 1

    entries: List[Dict[str, Any]] = json.loads(corr_json.read_text(encoding="utf-8"))

    # Prefer the per-pairing nested files when they exist.  Falling back
    # to the legacy top-level ``rank_comparison.csv`` only when there
    # are no pairing_*/ subdirectories closes a silent-data-loss bug:
    # when a v0.4 multi-pairing run is performed in a directory that
    # already holds a v0.3 single-pairing top-level CSV (no pairing_mode
    # column), the previous "if rank_cmp.exists() else nested" order
    # caused the loop below to label every legacy row as
    # ``pairing_mode = 'dimension'`` and therefore skip the topology /
    # random entries in ``correlation.json`` — they were left without
    # CI fields.  See benchmark/tests/test_paper_pipeline.py
    # ``TestRecomputeSpearmanCi``.
    nested_paths = sorted(rr_dir.glob("pairing_*/rank_comparison.csv"))
    if nested_paths:
        frames = []
        for path in nested_paths:
            frame = pd.read_csv(path)
            mode = path.parent.name[len("pairing_"):]
            if "pairing_mode" not in frame.columns:
                frame["pairing_mode"] = mode
            else:
                frame["pairing_mode"] = frame["pairing_mode"].fillna(mode)
            frames.append(frame)
        ranks = pd.concat(frames, ignore_index=True)
    elif rank_cmp.exists():
        ranks = pd.read_csv(rank_cmp)
    else:
        print(
            f"[err] neither {rank_cmp} nor any "
            f"{rr_dir}/pairing_*/rank_comparison.csv files found"
        )
        return 1
    if "pairing_mode" not in ranks.columns:
        modes = sorted({str(e["pairing_mode"]) for e in entries if "pairing_mode" in e})
        ranks["pairing_mode"] = modes[0] if len(modes) == 1 else "dimension"

    by_key = {}
    for entry in entries:
        mode = str(entry.get("pairing_mode", "dimension"))
        by_key[(mode, int(entry["budget_mult"]))] = entry

    for (mode, budget), sub in ranks.groupby(["pairing_mode", "budget_mult"]):
        mode = str(mode)
        b = int(budget)
        mode_seed = sum(ord(ch) for ch in mode)
        print(f"\n[pairing={mode}, budget = {b}d]  bootstrapping N={len(sub)} methods × n_boot={n_boot}")
        ci = bootstrap_rho_ci(
            sub["rank_analytical"].to_numpy(),
            sub["rank_physical"].to_numpy(),
            n_boot=n_boot,
            seed=seed + b + mode_seed,
        )
        print(
            f"  Spearman ρ = {ci['spearman_rho']:.3f}  "
            f"95% CI [{ci['spearman_rho_ci_lo']:.3f}, {ci['spearman_rho_ci_hi']:.3f}]"
            f"  (Fisher-z p={ci['spearman_p']:.3g})"
        )
        print(
            f"  Kendall  τ = {ci['kendall_tau']:.3f}  "
            f"95% CI [{ci['kendall_tau_ci_lo']:.3f}, {ci['kendall_tau_ci_hi']:.3f}]"
        )
        entry = by_key.setdefault((mode, b), {"pairing_mode": mode, "budget_mult": b})
        declared_n_methods = entry.get("n_methods")
        ranked_n_methods = ci.pop("n_methods")
        entry.update(ci)
        entry["n_ranked_methods_for_ci"] = int(ranked_n_methods)
        entry.update(_effective_case_counts(rr_dir, mode, b))
        if declared_n_methods is not None:
            entry["n_methods"] = declared_n_methods
        else:
            entry["n_methods"] = int(ranked_n_methods)

    out = [by_key[k] for k in sorted(by_key, key=lambda x: (x[0], x[1]))]
    corr_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[ok] wrote augmented correlation summary → {corr_json}")
    return 0


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--rr-dir",
        type=Path,
        default=ROOT / "benchmark" / "results" / "ranking_reversal",
        help="Directory containing rank_comparison.csv and correlation.json.",
    )
    p.add_argument("--n-boot", type=int, default=10_000, help="Bootstrap replicates.")
    p.add_argument("--seed", type=int, default=0, help="PRNG seed.")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    return recompute(args.rr_dir, n_boot=args.n_boot, seed=args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
