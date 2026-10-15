"""
benchmark.scripts.run_ranking_reversal
=======================================
The core scientific experiment of the Nature-track paper (Discovery 1):
**Are method rankings on analytical functions predictive of rankings
on physics-grounded engineering problems?**

Hypothesis (H1): Rankings systematically diverge — methods that excel
on smooth, low-dimensional analytical functions may underperform on
heterogeneous physical responses with threshold/discontinuity features.

Pairing modes (``--pairing-mode``)
----------------------------------
The analytical → physical pairing controls *which* physics case is
compared against *which* analytical function.  Three modes are
provided so the Discovery 1 conclusion can be cross-checked under
different pairing assumptions:

* ``dimension`` (default, original protocol).  Match by ``dim_input``
  — Branin (2D) ↔ TMD (3D), Hartmann (6D) ↔ wing-VLM (6D), etc.
  This is the most conservative pairing for the *power* of D1
  (matched-d controls for the dimension-driven cost difference).

* ``topology``.  Match by Jaccard similarity of
  ``CaseMeta.response_topology`` tags.  Each analytical case is paired
  with the physics case that shares the most topology labels
  (smooth_unimodal, threshold_jump, ...).  If multiple physics cases
  tie, the smallest ``|d_phys - d_anal|`` wins.  This pairing answers
  the reviewer concern *"would D1 still hold if we matched by
  difficulty rather than dimension?"*

* ``random``.  Random permutation of the physics cases (seed-controlled).
  Used as a *null distribution* — if D1 holds under both ``dimension``
  and ``topology`` but not under ``random``, the rank correspondence is
  driven by the structure of the pairing (which is what we hypothesise).

Outputs
-------
``benchmark/results/ranking_reversal/``
- ``analytical_results.csv``        — same N methods on 6 analytical functions
- ``physical_results.csv``          — re-run on a matched subset of physics cases
- ``rank_comparison.csv``           — side-by-side ranks (per pairing mode)
- ``correlation.json``              — Spearman ρ at each (budget, pairing) pair
- ``pairing_<mode>.json``           — explicit (analytical, physical) tuples
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import io

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError:
    pass

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from benchmark import (
    BenchmarkRunner,
    bo_method_pool,
    default_method_pool,
    extended_method_pool,
    full_method_pool,
    nature_pool,
)
from benchmark.analysis import pivot_metric, rank_methods
from benchmark.analytical import get_analytical
from benchmark.methods import mf_pool, sota_pool
from benchmark.stratified import bootstrap_rank_shift_ci
from scripts.prepare_figdata import validate_results_contract

PRIMARY_D1_ANALYSIS = "complete_case"

POOLS = {
    "default": default_method_pool,
    "extended": extended_method_pool,
    "full": full_method_pool,
    "bo": bo_method_pool,
    "nature": nature_pool,
    "sota": sota_pool,
    "mf": mf_pool,
}

# Six analytical test functions spanning d ∈ {2, 3, 5, 6, 8, 10}.
# This list is fixed; pairing to physics is parameterised below.
ANALYTICAL_CASES = [
    "branin_2d",         # d=2
    "ishigami_3d",       # d=3
    "friedman_5d",       # d=5
    "hartmann_6d",       # d=6
    "sobol_g_8d",        # d=8
    "wing_weight_10d",   # d=10
]

# ── Candidate physics cases per analytical d (for dimension pairing) ────
# Dimension pairing prefers an exact ``d`` match; if absent, we accept
# the closest ``d`` available from the pre-registered 34-case suite.
DIMENSION_PAIRING = {
    "branin_2d":          "tuned_mass_damper_3d",     # d=2 -> closest d=3
    "ishigami_3d":        "fin_heat_sink_4d",         # d=3 -> closest d=4 (smooth)
    "friedman_5d":        "cantilever_beam_5d",       # d=5 -> exact
    "hartmann_6d":        "wing_vlm_6d",              # d=6 -> exact
    "sobol_g_8d":         "aircraft_conceptual_8d",   # d=8 -> exact
    "wing_weight_10d":    "ten_bar_truss_10d",        # d=10 -> exact
}

# ── Topology tags assigned to each analytical function ──────────────────
# These mirror the CaseMeta.response_topology tagging convention used in
# the engineering testbed; sources are the seminal references for each
# function (Dixon-Szegö, Forrester et al.).
ANALYTICAL_TOPOLOGY = {
    "branin_2d":          ["smooth_multimodal", "smooth_unimodal"],
    "ishigami_3d":        ["strong_interaction", "sparse_effects"],
    "friedman_5d":        ["smooth_monotone", "strong_interaction"],
    "hartmann_6d":        ["smooth_multimodal"],
    "sobol_g_8d":         ["sparse_effects", "smooth_monotone"],
    "wing_weight_10d":    ["smooth_monotone", "smooth_unimodal"],
}


def _physics_topology(case_id: str) -> set[str]:
    """Return the set of response_topology tags for a physics case."""
    import doe_testbed as db
    case = db.get_case(case_id)
    tags = case.meta.response_topology or []
    if isinstance(tags, str):
        tags = [tags]
    return set(tags)


def _build_pairing(
    mode: str,
    seed: int,
    candidate_pool: Optional[List[str]] = None,
) -> dict:
    """Construct the analytical -> physical case pairing for the requested mode.

    Parameters
    ----------
    mode : 'dimension' | 'topology' | 'random'
    seed : int
        Random seed for the ``random`` pairing.
    candidate_pool : optional list of physics case_ids
        Restrict the physics pool to this list (default: all 34 cases).

    Returns
    -------
    dict
        Mapping ``analytical_id -> physical_id``, in the canonical
        ANALYTICAL_CASES order.
    """
    import doe_testbed as db

    if candidate_pool is None:
        candidate_pool = sorted(c.case_id for c in db.list_cases())

    if mode in {"topology", "random"} and len(candidate_pool) < len(ANALYTICAL_CASES):
        raise ValueError(
            f"{mode!r} pairing requires at least {len(ANALYTICAL_CASES)} "
            f"physical cases; got {len(candidate_pool)}"
        )

    if mode == "dimension":
        # Use the pre-registered pairing.  Validate the physics cases
        # are in the candidate pool, otherwise fall back to the
        # closest-d match from the candidate pool.
        out = {}
        for a in ANALYTICAL_CASES:
            preferred = DIMENSION_PAIRING.get(a)
            if preferred and preferred in candidate_pool:
                out[a] = preferred
            else:
                # Pick the candidate with closest dim_input
                d_anal = int(a.split("_")[-1].rstrip("d"))
                cands = [(abs(db.get_case(p).meta.dim_input - d_anal), p)
                         for p in candidate_pool]
                out[a] = sorted(cands)[0][1]
        return out

    if mode == "topology":
        # For each analytical case, pick the physics case with the
        # highest Jaccard similarity of topology tags; ties broken by
        # smallest |d_phys - d_anal|.
        out = {}
        used = set()
        for a in ANALYTICAL_CASES:
            tags_a = set(ANALYTICAL_TOPOLOGY.get(a, []))
            d_anal = int(a.split("_")[-1].rstrip("d"))

            scored = []
            for p in candidate_pool:
                if p in used:
                    continue
                tags_p = _physics_topology(p)
                if tags_a or tags_p:
                    jac = len(tags_a & tags_p) / max(len(tags_a | tags_p), 1)
                else:
                    jac = 0.0
                d_diff = abs(db.get_case(p).meta.dim_input - d_anal)
                scored.append((-jac, d_diff, p))
            scored.sort()
            out[a] = scored[0][2]
            used.add(scored[0][2])
        return out

    if mode == "random":
        rng = np.random.default_rng(seed)
        # Pick 6 distinct physics cases at random from the candidate pool
        chosen = list(rng.choice(candidate_pool, size=len(ANALYTICAL_CASES), replace=False))
        return dict(zip(ANALYTICAL_CASES, chosen))

    raise ValueError(f"unknown pairing mode {mode!r}; "
                     f"choose from 'dimension', 'topology', 'random'")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI options for the D1 analytical-vs-physics ranking-transfer analysis."""
    p = argparse.ArgumentParser(
        description="Ranking-reversal experiment: analytical vs physical.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--pool",
        choices=sorted(POOLS),
        default="nature",
        help="Method pool to use.",
    )
    p.add_argument("--n-repeats", type=int, default=30,
                   help="Number of independent repeats (≥5 for stable rank shift CI).")
    p.add_argument("--budgets", type=int, nargs="+", default=[10, 20],
                   help="Training budgets as multiples of d.")
    p.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help=(
            "Physical case_id candidate pool for analytical→physical pairings. "
            "Default/ALL uses all registered physics cases; smoke pipelines pass "
            "a small pool so D1 cannot accidentally expand to the full suite."
        ),
    )
    p.add_argument("--n-test", type=int, default=1000,
                   help="Reference test set size.")
    p.add_argument("--n-boot", type=int, default=1000,
                   help="Bootstrap replicates for rank-shift CI.")
    p.add_argument("--seed-strategy", choices=["case_repeat", "legacy"],
                   default="case_repeat",
                   help="Per-cell seed derivation; see benchmark.core.derive_seed.")
    p.add_argument(
        "--seed-start",
        type=int,
        default=1000,
        help="Seed salt / starting offset.",
    )
    p.add_argument(
        "--pairing-mode",
        choices=["dimension", "topology", "random", "all"],
        default="dimension",
        help=(
            "Pairing strategy between analytical functions and physics cases.  "
            "'all' runs all three modes back-to-back (recommended for the paper "
            "robustness section)."
        ),
    )
    p.add_argument(
        "--pairing-seed",
        type=int,
        default=2026,
        help="Seed used by the 'random' pairing mode (kept stable across runs).",
    )
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "benchmark" / "results" / "ranking_reversal",
                   help="Where to write CSV / JSON outputs.")
    p.add_argument(
        "--n-jobs", type=int, default=1,
        help=(
            "joblib worker count for the inner BenchmarkRunner.  Default 1 "
            "(serial) for legacy reproducibility; set to ≥ 8 on a Linux "
            "server for ~10× speed-up on the analytical + 3-pairing sweep."
        ),
    )
    p.add_argument(
        "--bo-n-jobs", type=int, default=None,
        help=(
            "Heavy-method (BoTorch SAASBO / qEI) worker cap.  Defaults to "
            "max(1, n_jobs // 4) inside BenchmarkRunner."
        ),
    )
    p.add_argument(
        "--mem-cap-gb", type=float, default=None,
        help="Soft RSS guard (GB).  Drops to serial + GC when exceeded.",
    )
    p.add_argument(
        "--skip-existing", action="store_true",
        help=(
            "Resume from a partial run.  Both the analytical-phase CSV "
            "(``<out>/analytical/results.csv``) and each pairing's physical-"
            "phase CSV (``<out>/pairing_<mode>/physical/results.csv``) are "
            "consulted; any (case, method, budget, repeat) tuple already "
            "present is skipped."
        ),
    )
    p.add_argument(
        "--timeout-s", type=float, default=600.0,
        help="Per-cell soft timeout for sklearn/static methods (seconds).",
    )
    p.add_argument(
        "--bo-timeout-s", type=float, default=1800.0,
        help="Per-cell soft timeout for BoTorch methods (seconds).",
    )
    p.add_argument(
        "--ref-timeout-s", type=float, default=600.0,
        help="Reference-set construction timeout (seconds).",
    )
    p.add_argument(
        "--reuse-from-csv", type=Path, default=None,
        help=(
            "Skip both phases' BenchmarkRunner and reuse cells from an "
            "existing results CSV (e.g. the headline full_benchmark sweep). "
            "Analytical cases are filtered by ANALYTICAL_CASES; physical "
            "cases are filtered per pairing mode. Drastically faster than "
            "re-running the full sweep when the headline data is already "
            "available."
        ),
    )
    p.add_argument(
        "--reuse-analytical-from", type=Path, default=None,
        help=(
            "Optional override of --reuse-from-csv for the analytical "
            "phase only.  Useful when analytical cases are not present in "
            "the headline CSV (e.g. a separate analytical sweep CSV)."
        ),
    )
    return p.parse_args(argv)


def _resolve_candidate_pool(case_args: Optional[List[str]]) -> Optional[List[str]]:
    """Normalize ``--cases`` into the physical candidate pool used by D1."""
    if not case_args or case_args == ["ALL"]:
        return None

    import doe_testbed as db

    available = {c.case_id for c in db.list_cases()}
    requested = list(dict.fromkeys(case_args))
    unknown = sorted(set(requested) - available)
    if unknown:
        raise SystemExit(f"Unknown physics case_id(s) for --cases: {', '.join(unknown)}")
    if not requested:
        raise SystemExit("--cases must name at least one physical case")
    return requested


def _validate_pairing_modes_for_candidate_pool(
    modes: List[str],
    candidate_pool: Optional[List[str]],
) -> None:
    """Fail early when the requested D1 pairing cannot be formed."""
    if (
        candidate_pool is not None
        and any(m in {"topology", "random"} for m in modes)
        and len(candidate_pool) < len(ANALYTICAL_CASES)
    ):
        raise SystemExit(
            "Topology/random D1 pairings require at least "
            f"{len(ANALYTICAL_CASES)} physical cases; got {len(candidate_pool)}. "
            "Use --pairing-mode dimension for smoke-scale probes."
        )


def _hydrate_full_results(df_new: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    """Return the full history of a BenchmarkRunner, merging in-memory + streamed.

    ``BenchmarkRunner.run()`` returns only the cells it actually computed
    in this invocation.  When ``--skip-existing`` is active and the run
    is a full resume, ``df_new`` is empty.  The analysis phase needs the
    full history, which is always present in the streamed CSV at
    ``output_dir/results.csv``.  This helper prefers the CSV when it
    exists and falls back to the in-memory frame otherwise.
    """
    if csv_path.exists():
        try:
            df_full = pd.read_csv(csv_path)
            if len(df_full) >= len(df_new):
                return df_full
        except Exception:
            pass
    return df_new


def _read_reusable_results_csv(path: Path) -> pd.DataFrame:
    """Read and validate an existing sweep CSV before D1 reuse filtering."""
    df = pd.read_csv(path, low_memory=False)
    cell_key_columns = ["case_id", "method", "budget_mult", "repeat"]
    if set(cell_key_columns).issubset(df.columns):
        duplicate_cells = df.duplicated(cell_key_columns, keep=False)
        if bool(duplicate_cells.any()):
            examples = (
                df.loc[duplicate_cells, cell_key_columns]
                .drop_duplicates()
                .head(10)
                .to_dict("records")
            )
            raise AssertionError(
                "results.csv contains duplicate "
                "(case_id, method, budget_mult, repeat) cells: "
                f"{examples}"
            )
    validate_results_contract(df)
    return df


def _run_one_pairing(
    pairing_mode: str,
    pairing: dict,
    methods: list,
    budgets: list,
    n_repeats: int,
    n_test: int,
    n_boot: int,
    seed_strategy: str,
    out_dir: Path,
    res_a_df: pd.DataFrame,
    runner_kwargs: Optional[dict] = None,
    reuse_from_csv: Optional[Path] = None,
) -> dict:
    """Run the physical-case half + analyse the rank correspondence for one pairing."""
    physical_cases = list(dict.fromkeys(pairing.values()))
    pairing_dir = out_dir / f"pairing_{pairing_mode}"
    pairing_dir.mkdir(parents=True, exist_ok=True)
    runner_kwargs = dict(runner_kwargs or {})

    # Persist the pairing JSON so reviewers can audit it
    with open(pairing_dir / "pairing.json", "w") as f:
        json.dump(pairing, f, indent=2)

    print(f"\n[{pairing_mode}] Pairing:")
    for a, p in pairing.items():
        print(f"    {a:25s}  ↔  {p}")

    if reuse_from_csv is not None:
        # Fast path: filter the headline CSV to this pairing's physical
        # cases instead of re-running the BenchmarkRunner.
        src = Path(reuse_from_csv)
        print(f"\n[{pairing_mode}] Reusing {len(physical_cases)} physical cases from {src}")
        t0 = time.perf_counter()
        full_df = _read_reusable_results_csv(src)
        df_p = full_df[full_df["case_id"].isin(physical_cases)].copy()
        (pairing_dir / "physical").mkdir(parents=True, exist_ok=True)
        df_p.to_csv(pairing_dir / "physical" / "results.csv", index=False)
        df_p.to_csv(pairing_dir / "physical_results.csv", index=False)
        print(
            f"  → physical phase reused {len(df_p)} cells in "
            f"{(time.perf_counter()-t0):.1f} s"
        )
    else:
        # Avoid duplicate physics-case work: if a case is in physical_cases,
        # share the BenchmarkRunner output but tag it for this pairing.
        print(f"\n[{pairing_mode}] Running {len(methods)} methods on {len(physical_cases)} physics cases")
        t0 = time.perf_counter()
        runner_p = BenchmarkRunner(
            case_ids=physical_cases,
            methods=methods,
            budgets=budgets,
            n_repeats=n_repeats,
            n_test=n_test,
            seed_strategy=seed_strategy,
            output_dir=pairing_dir / "physical",
            verbose=True,
            **runner_kwargs,
        )
        res_p = runner_p.run()
        # When --skip-existing replays a completed run, `res_p.df` only holds
        # the *newly* computed rows (empty on full resume).  The streamed
        # CSV at ``output_dir/results.csv`` always holds the full history,
        # so re-hydrate from it before the rank analysis runs.
        df_p = _hydrate_full_results(res_p.df, pairing_dir / "physical" / "results.csv")
        df_p.to_csv(pairing_dir / "physical_results.csv", index=False)
        print(f"  → physical phase done in {(time.perf_counter()-t0)/60:.1f} min")

    # ── Map analytical -> physical ranks per the pairing ──
    rows = []
    summary_per_budget = {}
    for budget in budgets:
        pivot_a = pivot_metric(res_a_df, "nrmse", budget_value=budget)
        pivot_p = pivot_metric(df_p, "nrmse", budget_value=budget)
        # Per-case rank vector for analytical, then average over the
        # 6 paired analytical cases to get a method-rank vector.
        # (This preserves the existing aggregation: rank_methods averages
        # over all cases in the pivot.)
        ranks_a = rank_methods(pivot_a, lower_is_better=True)
        ranks_p = rank_methods(pivot_p, lower_is_better=True)
        common = [m for m in ranks_a.index if m in ranks_p.index]
        ra, rp = ranks_a.loc[common], ranks_p.loc[common]

        rho, p_rho = spearmanr(ra.values, rp.values)
        tau, p_tau = kendalltau(ra.values, rp.values)
        shift = float(np.mean(np.abs(ra.values - rp.values)))

        for m in common:
            rows.append({
                "pairing_mode": pairing_mode,
                "budget_mult": int(budget),
                "method": m,
                "rank_analytical": float(ra[m]),
                "rank_physical": float(rp[m]),
                "rank_shift": float(rp[m] - ra[m]),
                "analysis_type": PRIMARY_D1_ANALYSIS,
                "is_primary": True,
            })
        summary_per_budget[int(budget)] = {
            "spearman_rho": float(rho),
            "spearman_p": float(p_rho),
            "kendall_tau": float(tau),
            "kendall_p": float(p_tau),
            "mean_abs_rank_shift": shift,
            "analysis_type": PRIMARY_D1_ANALYSIS,
            "is_primary": True,
        }

        print(
            f"  [{pairing_mode}] Budget = {budget}d:  ρ = {rho:.3f}  (p={p_rho:.3g})"
            f"   τ = {tau:.3f}  mean|Δrank| = {shift:.2f}"
        )

    rank_df = pd.DataFrame(rows)
    rank_df.to_csv(pairing_dir / "rank_comparison.csv", index=False)

    # Bootstrap CIs on the rank-shift per (method, budget)
    shift_frames = {}
    for budget in budgets:
        shift_frames[int(budget)] = bootstrap_rank_shift_ci(
            df_a=res_a_df, df_b=df_p,
            metric="nrmse", lower_is_better=True,
            budget_value=budget, n_boot=n_boot, seed=0,
        ).assign(budget_mult=int(budget), pairing_mode=pairing_mode)
    shift_all = pd.concat(shift_frames.values(), ignore_index=True)
    shift_all.to_csv(pairing_dir / "rank_shift_bootstrap_ci.csv", index=False)

    return {
        "pairing_mode": pairing_mode,
        "pairing": pairing,
        "rank_comparison": rows,
        "summary_per_budget": summary_per_budget,
    }


def main(argv: Optional[List[str]] = None) -> int:
    """Run the registered D1 analysis: score methods on the analytical test functions, correlate with the frozen physics-case rankings, and write the correlation artifacts."""
    args = parse_args(argv)
    methods = POOLS[args.pool]()
    budgets = list(args.budgets)
    n_repeats = int(args.n_repeats)
    n_test = int(args.n_test)
    n_boot = int(args.n_boot)
    candidate_pool = _resolve_candidate_pool(args.cases)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve which pairing mode(s) to run
    if args.pairing_mode == "all":
        modes = ["dimension", "topology", "random"]
    else:
        modes = [args.pairing_mode]
    _validate_pairing_modes_for_candidate_pool(modes, candidate_pool)

    print("=" * 70)
    print("Ranking-Reversal Experiment: analytical vs physical")
    print("=" * 70)
    print(f"Pool          : {args.pool}")
    print(f"Methods       : {len(methods)} ({', '.join(m.name for m in methods)})")
    print(f"Budgets       : {budgets}")
    print(f"Repeats       : {n_repeats}")
    print(f"Pairing modes : {modes}")
    print(f"Physical cases: {'ALL' if candidate_pool is None else candidate_pool}")
    print(f"Seed          : strategy={args.seed_strategy}, salt={args.seed_start}, pairing_seed={args.pairing_seed}")
    print(f"n_jobs        : {args.n_jobs} (BO_n_jobs={args.bo_n_jobs}, mem_cap={args.mem_cap_gb})")
    print(f"skip_existing : {args.skip_existing}")
    print(f"timeouts      : per-cell={args.timeout_s}s  bo={args.bo_timeout_s}s  ref={args.ref_timeout_s}s")
    if args.reuse_from_csv is not None:
        print(f"reuse_from_csv: {args.reuse_from_csv}  (fast path: skip BenchmarkRunner)")
    print(f"Output        : {out_dir}")
    print("=" * 70)

    # Common BenchmarkRunner kwargs threaded through both phases
    runner_kwargs = dict(
        n_jobs=int(args.n_jobs),
        bo_n_jobs=args.bo_n_jobs,
        mem_cap_gb=args.mem_cap_gb,
        skip_existing=bool(args.skip_existing),
        timeout_s=float(args.timeout_s),
        bo_timeout_s=float(args.bo_timeout_s),
        ref_timeout_s=float(args.ref_timeout_s),
        seed_start=int(args.seed_start),
    )

    # ── Phase 1: analytical (shared across all pairings) ──
    if args.reuse_from_csv is not None or args.reuse_analytical_from is not None:
        # Fast path: reuse the headline sweep's CSV instead of re-running
        # the analytical BenchmarkRunner.  Filter to ANALYTICAL_CASES only.
        src = Path(args.reuse_analytical_from or args.reuse_from_csv)
        print(f"\n[Phase 1/2]  Reusing analytical cases from {src}")
        t0 = time.perf_counter()
        full_df = _read_reusable_results_csv(src)
        res_a_df = full_df[full_df["case_id"].isin(ANALYTICAL_CASES)].copy()
        (out_dir / "analytical").mkdir(parents=True, exist_ok=True)
        res_a_df.to_csv(out_dir / "analytical" / "results.csv", index=False)
        res_a_df.to_csv(out_dir / "analytical_results.csv", index=False)
        print(
            f"  → analytical phase reused {len(res_a_df)} cells in "
            f"{(time.perf_counter()-t0):.1f} s"
        )
    else:
        print(f"\n[Phase 1/2]  Running {len(methods)} methods on "
              f"{len(ANALYTICAL_CASES)} analytical functions")
        t0 = time.perf_counter()
        runner_a = BenchmarkRunner(
            case_ids=ANALYTICAL_CASES,
            methods=methods,
            budgets=budgets,
            n_repeats=n_repeats,
            n_test=n_test,
            seed_strategy=args.seed_strategy,
            output_dir=out_dir / "analytical",
            verbose=True,
            case_loader=get_analytical,
            **runner_kwargs,
        )
        res_a = runner_a.run()
        res_a_df = _hydrate_full_results(res_a.df, out_dir / "analytical" / "results.csv")
        res_a_df.to_csv(out_dir / "analytical_results.csv", index=False)
        print(f"  → analytical phase done in {(time.perf_counter()-t0)/60:.1f} min")

    # ── Phase 2: physical, per pairing mode ──
    summaries = []
    for mode in modes:
        pairing = _build_pairing(mode, seed=args.pairing_seed, candidate_pool=candidate_pool)
        out = _run_one_pairing(
            pairing_mode=mode,
            pairing=pairing,
            methods=methods,
            budgets=budgets,
            n_repeats=n_repeats,
            n_test=n_test,
            n_boot=n_boot,
            seed_strategy=args.seed_strategy,
            out_dir=out_dir,
            res_a_df=res_a_df,
            runner_kwargs=runner_kwargs,
            reuse_from_csv=args.reuse_from_csv,
        )
        summaries.append(out)

    # ── Aggregate cross-pairing comparison summary ──
    corr_summary = []
    for s in summaries:
        for budget, vals in s["summary_per_budget"].items():
            corr_summary.append({
                "pairing_mode": s["pairing_mode"],
                "budget_mult": int(budget),
                **vals,
                "n_repeats": n_repeats,
                "method_pool": args.pool,
                "n_methods": len(methods),
                "seed_start": int(args.seed_start),
            })
    with open(out_dir / "correlation.json", "w") as f:
        json.dump(corr_summary, f, indent=2)
    print(f"\nSaved correlation summary: {out_dir/'correlation.json'}")

    # Print a final summary table comparing modes
    if len(modes) > 1:
        print("\n" + "=" * 70)
        print("Cross-pairing summary (Spearman ρ at each budget):")
        print("=" * 70)
        print(f"  {'mode':<12s}  " + "  ".join(f"b={b:<2d}d" for b in budgets))
        for s in summaries:
            row = [f"{s['summary_per_budget'][int(b)]['spearman_rho']:>+5.3f}" for b in budgets]
            print(f"  {s['pairing_mode']:<12s}  " + "  ".join(f"{r:>5s}  " for r in row))
        print()
        print("Interpretation:")
        print("  * If ρ < 0.7 under BOTH 'dimension' and 'topology' but rises under 'random',")
        print("    Discovery 1's rank-divergence finding is robust to pairing choice.")
        print("  * If ρ stays similar across all three modes, the headline is rank-")
        print("    invariant in the methods themselves — equally valid scientifically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
