"""Case-axis bootstrap and split-half reliability for the registered D1 rank correlation.

Why this script exists
----------------------
``correlation.json`` reports Spearman rho = 0.914 between the analytical rank
vector and the physical rank vector at the 10d budget, with a 95 % CI obtained
by resampling *methods*.  That interval answers "how stable is rho if we had
drawn a different method pool?", but it says nothing about the other sampling
axis: each rank vector is itself an average over a small panel of cases
(6 analytical functions, 4 complete physical cases).

Two consequences follow, and both are reported here so the manuscript can state
them instead of leaving them implicit:

* the *case* axis is a second, independent source of sampling variability for
  rho, so a case-axis bootstrap is the complementary interval; and
* if each rank vector is only moderately reliable across its own cases, then
  even a perfect cross-regime relationship cannot reach rho = 1.  The
  Spearman-Brown split-half reliability of each side, combined through the
  attenuation bound sqrt(rel_a * rel_p), gives that ceiling.

Everything is recomputed from the frozen archived sweep; no number is
hand-entered.  Usage:

    python scripts/derive_d1_case_axis_reliability.py
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from benchmark.analysis import pivot_metric, rank_methods

RR_DIR = ROOT / "benchmark" / "results" / "ranking_reversal"
PRIMARY_PAIRING = "dimension"
PRIMARY_BUDGET = 10
METRIC = "nrmse"

B_CASE = 10_000
B_SPLIT_HALF = 1_000
SEED = 1000
RHO_REFERENCE = 0.70


def _rank_vector(pivot: pd.DataFrame) -> pd.Series:
    """Mean per-case rank per method, on the complete-case policy."""
    return rank_methods(pivot, lower_is_better=True)


def _complete_cases(pivot: pd.DataFrame) -> pd.DataFrame:
    """Rows that survive the registered complete-case policy.

    ``rank_methods`` drops any case with a missing method value before ranking.
    Resampling must therefore draw from exactly those rows, otherwise the
    bootstrap would repeatedly inject rows the estimator never uses.
    """
    return pivot.dropna(how="any")


def _rho(ranks_a: pd.Series, ranks_b: pd.Series) -> float:
    common = [m for m in ranks_a.index if m in ranks_b.index]
    if len(common) < 3:
        return float("nan")
    ra = ranks_a.loc[common].to_numpy(dtype=float)
    rb = ranks_b.loc[common].to_numpy(dtype=float)
    if not np.all(np.isfinite(ra)) or not np.all(np.isfinite(rb)):
        return float("nan")
    if np.unique(ra).size < 2 or np.unique(rb).size < 2:
        return float("nan")
    value, _ = spearmanr(ra, rb)
    return float(value)


def case_axis_bootstrap(
    pivot_a: pd.DataFrame,
    pivot_b: pd.DataFrame,
    *,
    n_boot: int = B_CASE,
    seed: int = SEED,
) -> dict:
    """Resample the *case* rows of both rank vectors with replacement."""
    rng = np.random.default_rng(seed)
    rows_a = pivot_a.index.to_numpy()
    rows_b = pivot_b.index.to_numpy()
    draws = np.empty(n_boot, dtype=float)
    for k in range(n_boot):
        sa = pivot_a.loc[rng.choice(rows_a, size=len(rows_a), replace=True)]
        sb = pivot_b.loc[rng.choice(rows_b, size=len(rows_b), replace=True)]
        draws[k] = _rho(_rank_vector(sa), _rank_vector(sb))
    finite = draws[np.isfinite(draws)]
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return {
        "n_boot": int(n_boot),
        "bootstrap_seed": int(seed),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n_finite_draws": int(finite.size),
        "median": float(np.median(finite)),
        "prob_below_reference": float((finite < RHO_REFERENCE).mean()),
        "reference_rho": float(RHO_REFERENCE),
    }


def split_half_reliability(
    pivot: pd.DataFrame,
    *,
    n_boot: int = B_SPLIT_HALF,
    seed: int = SEED,
) -> dict:
    """Spearman-Brown split-half reliability of a rank vector.

    Cases are split into two halves; each half yields its own mean-rank vector;
    the two halves are correlated and corrected with the Spearman-Brown
    prophecy formula ``r_sb = 2r / (1 + r)``.  For small case counts every
    distinct split is enumerated; otherwise ``n_boot`` random splits are drawn.
    """
    n_cases = int(pivot.shape[0])
    half = n_cases // 2
    if half == 0:
        return {"n_cases": n_cases, "reliability": float("nan"),
                "n_splits": 0, "scheme": "undefined"}

    indices = list(range(n_cases))
    if n_cases <= 12:
        splits = [c for c in combinations(indices, half) if 0 in c]
        scheme = "exhaustive"
    else:
        rng = np.random.default_rng(seed)
        splits = [tuple(rng.choice(indices, size=half, replace=False))
                  for _ in range(n_boot)]
        scheme = "random"

    values = []
    for left in splits:
        right = tuple(i for i in indices if i not in left)
        if not right:
            continue
        ra = _rank_vector(pivot.iloc[list(left)])
        rb = _rank_vector(pivot.iloc[list(right)])
        r = _rho(ra, rb)
        if np.isfinite(r) and r > -1.0:
            values.append(2.0 * r / (1.0 + r))
    values = np.asarray(values, dtype=float)
    return {
        "n_cases": n_cases,
        "half_size": int(half),
        "n_splits": int(values.size),
        "scheme": scheme,
        "reliability": float(values.mean()) if values.size else float("nan"),
        "reliability_sd": float(values.std(ddof=1)) if values.size > 1 else float("nan"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rr-dir", type=Path, default=RR_DIR)
    parser.add_argument("--pairing", default=PRIMARY_PAIRING)
    parser.add_argument("--budget", type=int, default=PRIMARY_BUDGET)
    parser.add_argument("--n-boot", type=int, default=B_CASE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    analytical_csv = args.rr_dir / "analytical" / "results.csv"
    physical_csv = args.rr_dir / f"pairing_{args.pairing}" / "physical" / "results.csv"
    for path in (analytical_csv, physical_csv):
        if not path.exists():
            print(f"[d1-case-axis:error] missing {path}", file=sys.stderr)
            return 2

    df_a = pd.read_csv(analytical_csv, low_memory=False)
    df_p = pd.read_csv(physical_csv, low_memory=False)
    pivot_a = pivot_metric(df_a, METRIC, budget_value=args.budget)
    pivot_p = pivot_metric(df_p, METRIC, budget_value=args.budget)

    complete_a = _complete_cases(pivot_a)
    complete_p = _complete_cases(pivot_p)
    ra, rb = _rank_vector(pivot_a), _rank_vector(pivot_p)
    rho_point = _rho(ra, rb)

    # Cross-check against the registered summary so a silent drift is loud.
    corr_path = args.rr_dir / "correlation.json"
    rho_registered = None
    if corr_path.exists():
        for row in json.loads(corr_path.read_text(encoding="utf-8")):
            if (row.get("pairing_mode") == args.pairing
                    and int(row.get("budget_mult", -1)) == args.budget):
                rho_registered = float(row["spearman_rho"])
                break

    rel_a = split_half_reliability(complete_a, seed=args.seed)
    rel_p = split_half_reliability(complete_p, seed=args.seed)
    if np.isfinite(rel_a["reliability"]) and np.isfinite(rel_p["reliability"]):
        ceiling = float(np.sqrt(max(rel_a["reliability"], 0.0)
                                * max(rel_p["reliability"], 0.0)))
    else:
        ceiling = float("nan")

    payload = {
        "pairing_mode": args.pairing,
        "budget_mult": int(args.budget),
        "metric": METRIC,
        "n_methods": int(len(ra)),
        "n_analytical_cases": int(pivot_a.shape[0]),
        "n_physical_cases": int(pivot_p.shape[0]),
        "spearman_rho": rho_point,
        "spearman_rho_registered": rho_registered,
        "n_analytical_cases_complete": int(complete_a.shape[0]),
        "n_physical_cases_complete": int(complete_p.shape[0]),
        "case_axis_bootstrap": case_axis_bootstrap(
            complete_a, complete_p, n_boot=args.n_boot, seed=args.seed
        ),
        "split_half_analytical": rel_a,
        "split_half_physical": rel_p,
        "reliability_ceiling": ceiling,
        "source_analytical": str(analytical_csv.relative_to(ROOT)).replace("\\", "/"),
        "source_physical": str(physical_csv.relative_to(ROOT)).replace("\\", "/"),
        "script": "scripts/derive_d1_case_axis_reliability.py",
    }

    out = args.rr_dir / "d1_case_axis_reliability.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items()
                      if k not in ("source_analytical", "source_physical")}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
