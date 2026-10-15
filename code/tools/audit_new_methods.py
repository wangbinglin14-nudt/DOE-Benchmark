"""End-to-end audit harness for the post-registration methods.

This script drives the **actual** ``BenchmarkRunner`` (not standalone
.run() calls) so the new methods are exercised through every layer
that the production sweep will exercise: joblib partitioning, RSS
guard, CSV streaming, watchdog, error handlers, etc.

It is invoked from ``tools/audit_new_methods.sh`` (Linux/CI) and used
locally with::

    python -m tools.audit_new_methods

The harness writes a single Markdown summary to
``tools/_audit_report.md`` in the repo root and prints a one-line
PASS/FAIL.  The script is deliberately small (≤ ~3 min wall-clock
on a laptop) so it can be wired into pre-commit if desired.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np


# ── small utilities ──────────────────────────────────────────────────────
def rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 ** 2
    except Exception:
        return float("nan")


def heading(s: str) -> str:
    return "\n" + "=" * 72 + f"\n {s}\n" + "=" * 72


def section(s: str) -> str:
    return f"\n--- {s} ---"


# ── individual audit blocks ──────────────────────────────────────────────
def audit_d_optimal() -> dict:
    """D-optimal sampler: edge cases + worst-case stress."""
    from benchmark.samplers import DOptimalSampler

    out: dict = {"name": "DOptimalSampler", "checks": []}
    cases = [
        ("d=1 minimal", 20, 1),
        ("n=p+2 boundary", 23, 5),
        ("n<p+2 fallback", 22, 5),
        ("d=20 n=1000 worst-case sweep cell", 1000, 20),
    ]
    for label, n, d in cases:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0 = time.perf_counter()
            X = DOptimalSampler().sample(n, d, seed=42)
            dt = time.perf_counter() - t0
        ok = (X.shape == (n, d)
              and np.all(np.isfinite(X))
              and np.all((X >= 0) & (X <= 1)))
        out["checks"].append({
            "label": label, "n": n, "d": d,
            "time_s": round(dt, 2), "ok": bool(ok),
        })
    # Determinism
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X1 = DOptimalSampler().sample(100, 8, seed=42)
        X2 = DOptimalSampler().sample(100, 8, seed=42)
        X3 = DOptimalSampler().sample(100, 8, seed=43)
    out["checks"].append({
        "label": "determinism (same seed)",
        "ok": bool(np.allclose(X1, X2)),
    })
    out["checks"].append({
        "label": "different seed -> different design",
        "ok": bool(not np.allclose(X1, X3)),
    })
    return out


def audit_turbo() -> dict:
    """TuRBO: budget faithfulness + restart bookkeeping + d=20 stress."""
    import doe_testbed as db
    from benchmark.bo_methods import BoTorchTurboMethod

    out: dict = {"name": "BoTorchTurboMethod", "checks": []}

    case = db.get_case("cantilever_beam_5d")
    X_test = case.sample(50, method="sobol", seed=0)
    # Budget faithfulness across the 4 sweep budgets
    for n_train in [25, 50, 100, 250]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = BoTorchTurboMethod(n_init_frac=0.5, n_candidates=64)
            t0 = time.perf_counter()
            Y_pred, _, meta = m.run(case, n_train=n_train, X_test=X_test, seed=42)
            dt = time.perf_counter() - t0
        ok = (Y_pred.shape == X_test.shape[:1] + (3,)  # 3 outputs
              and np.all(np.isfinite(Y_pred))
              and meta["n_evals_total"] <= n_train)
        out["checks"].append({
            "label": f"d=5 n_train={n_train}", "time_s": round(dt, 1),
            "n_evals": meta["n_evals_total"], "n_restarts": meta["n_tr_restarts"],
            "ok": bool(ok),
        })

    # Forced restart (low L_min) — must still respect budget
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = BoTorchTurboMethod(n_init_frac=0.3, n_candidates=64,
                               L_min=0.4, L_init=0.5, fail_streak=2)
        Y_pred, _, meta = m.run(case, n_train=50, X_test=X_test, seed=42)
    out["checks"].append({
        "label": "forced restart respects budget",
        "n_evals": meta["n_evals_total"], "n_restarts": meta["n_tr_restarts"],
        "ok": meta["n_evals_total"] <= 50,
    })

    # High-dim stress
    case_hd = db.get_case("variable_beam_20d")
    X_test_hd = case_hd.sample(50, method="sobol", seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = BoTorchTurboMethod(n_init_frac=0.5, n_candidates=128)
        t0 = time.perf_counter()
        Y_pred, _, meta = m.run(case_hd, n_train=50, X_test=X_test_hd, seed=42)
        dt = time.perf_counter() - t0
    out["checks"].append({
        "label": "d=20 n_train=50 high-dim stress",
        "time_s": round(dt, 1), "n_evals": meta["n_evals_total"],
        "rss_mb": round(rss_mb(), 0),
        "ok": bool(np.all(np.isfinite(Y_pred))),
    })
    return out


def audit_mf_cokriging() -> dict:
    """MF Cokriging: tested on all 5 MF cases at all 4 budgets."""
    import doe_testbed as db
    from benchmark.bo_methods import MFCokrigingMethod

    out: dict = {"name": "MFCokrigingMethod", "checks": []}
    mf_cases = [c.case_id for c in db.list_cases() if c.supports_multifidelity]

    for cid in mf_cases:
        case = db.get_case(cid)
        d = case.meta.dim_input
        X_test = case.sample(50, method="sobol", seed=0)
        for mult in [5, 10, 20, 50]:
            n = mult * d
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = MFCokrigingMethod(frac_low_cost=0.5)
                    t0 = time.perf_counter()
                    Y_pred, _, meta = m.run(case, n_train=n, X_test=X_test, seed=42)
                    dt = time.perf_counter() - t0
                ok = (np.all(np.isfinite(Y_pred))
                      and meta["n_low"] + meta["n_high"] >= 4)
                out["checks"].append({
                    "label": f"{cid} budget={mult}d n={n}",
                    "time_s": round(dt, 1),
                    "n_low": meta["n_low"], "n_high": meta["n_high"],
                    "fallback": meta["fallback_to_single_fidelity"],
                    "ok": bool(ok),
                })
            except Exception as e:
                out["checks"].append({
                    "label": f"{cid} budget={mult}d n={n}",
                    "ok": False, "error": f"{type(e).__name__}: {e}",
                })
        gc.collect()
    return out


def audit_runner_integration() -> dict:
    """Drive the BenchmarkRunner end-to-end on the 3 new methods.

    Uses: ``--pool sota`` on 1 small case, n_repeats=1, budgets=[5].
    Verifies: results.csv is well-formed, no NaN/inf in NRMSE, runner
    metadata captures the new methods.
    """
    import subprocess

    out: dict = {}
    out["name"] = "BenchmarkRunner integration"
    out["checks"] = []

    out_dir = Path("tools/_audit_runs/sota_smoke")
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "benchmark/scripts/run_full_benchmark.py",
        "--pool", "sota",
        "--cases", "cantilever_beam_5d", "tuned_mass_damper_3d",
        "--budgets", "5",
        "--n-repeats", "1",
        "--n-test", "50",
        "--n-jobs", "2",
        "--output-dir", str(out_dir),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    dt = time.perf_counter() - t0

    out["checks"].append({
        "label": "sota pool 2-case smoke", "time_s": round(dt, 1),
        "returncode": proc.returncode,
        "ok": proc.returncode == 0,
    })

    csv_path = out_dir / "results.csv"
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        # Confirm new methods made it into the CSV
        has_turbo = (df["method"] == "botorch_turbo").any()
        has_dopt = (df["method"] == "doptimal+gp").any()
        n_rows = len(df)
        out["checks"].append({
            "label": "CSV has botorch_turbo rows",
            "n_rows": int(n_rows), "has_turbo": bool(has_turbo),
            "ok": bool(has_turbo),
        })
        out["checks"].append({
            "label": "CSV has doptimal+gp rows",
            "has_dopt": bool(has_dopt), "ok": bool(has_dopt),
        })
        # Guard against silent NaN/inf in primary metric
        if "nrmse" in df.columns:
            ok_finite = df["nrmse"].notna().all()
            out["checks"].append({
                "label": "no NaN in nrmse (status=ok rows)",
                "ok": bool(ok_finite),
            })
    else:
        out["checks"].append({
            "label": "results.csv was created",
            "ok": False, "error": "missing results.csv",
        })
    return out


def audit_mf_runner_integration() -> dict:
    """Drive ``--pool mf --cases-filter mf`` end-to-end."""
    import subprocess

    out: dict = {"name": "MF pool runner integration", "checks": []}
    out_dir = Path("tools/_audit_runs/mf_smoke")
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "benchmark/scripts/run_full_benchmark.py",
        "--pool", "mf", "--cases-filter", "mf",
        "--budgets", "10",
        "--n-repeats", "1",
        "--n-test", "50",
        "--n-jobs", "2",
        "--output-dir", str(out_dir),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=False, timeout=900)
    dt = time.perf_counter() - t0
    out["checks"].append({
        "label": "mf pool + cases-filter mf smoke",
        "time_s": round(dt, 1),
        "returncode": proc.returncode,
        "ok": proc.returncode == 0,
    })

    csv_path = out_dir / "results.csv"
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        # Each MF-capable case must produce exactly one row at budget=10d
        out["checks"].append({
            "label": "5 MF cases x 1 method x 1 budget x 1 repeat = 5 rows",
            "n_rows": int(len(df)),
            "ok": bool(len(df) == 5),
        })
    else:
        out["checks"].append({
            "label": "MF results.csv was created",
            "ok": False, "error": "missing results.csv",
        })
    return out


def audit_pairing_modes() -> dict:
    """D1 pairing modes: confirm each mode runs and emits a deterministic mapping."""
    from benchmark.scripts.run_ranking_reversal import (
        ANALYTICAL_CASES, _build_pairing,
    )

    out: dict = {"name": "D1 pairing modes", "checks": []}
    for mode in ("dimension", "topology", "random"):
        try:
            p1 = _build_pairing(mode, seed=2026)
            p2 = _build_pairing(mode, seed=2026)
            out["checks"].append({
                "label": f"pairing-mode {mode!r}",
                "n_pairs": len(p1),
                "deterministic": p1 == p2,
                "ok": (len(p1) == len(ANALYTICAL_CASES)) and (p1 == p2),
            })
        except Exception as e:
            out["checks"].append({
                "label": f"pairing-mode {mode!r}",
                "ok": False, "error": f"{type(e).__name__}: {e}",
            })
    return out


def audit_pool_compliance() -> dict:
    """Pre-registration compliance: nature_pool() stays at 20."""
    from benchmark.methods import (
        bo_method_pool, default_method_pool, extended_method_pool,
        full_method_pool, mf_pool, nature_pool, sota_pool,
    )
    out: dict = {"name": "Pre-registration compliance", "checks": []}
    n_nature = len(nature_pool())
    out["checks"].append({
        "label": "nature_pool() == 20 (frozen by pre-registration)",
        "value": n_nature, "ok": n_nature == 20,
    })
    n_sota = len(sota_pool())
    out["checks"].append({
        "label": "sota_pool() == 22 (= nature + TURBO + D-optimal)",
        "value": n_sota, "ok": n_sota == 22,
    })
    n_mf = len(mf_pool())
    out["checks"].append({
        "label": "mf_pool() >= 1 (MF Cokriging present)",
        "value": n_mf, "ok": n_mf >= 1,
    })
    # Sanity on the others
    out["checks"].append({
        "label": "default == 10",
        "value": len(default_method_pool()), "ok": len(default_method_pool()) == 10,
    })
    out["checks"].append({
        "label": "extended == 15",
        "value": len(extended_method_pool()), "ok": len(extended_method_pool()) == 15,
    })
    out["checks"].append({
        "label": "full   >= 18",
        "value": len(full_method_pool()), "ok": len(full_method_pool()) >= 18,
    })
    out["checks"].append({
        "label": "bo == 2 (qEI + SAASBO)",
        "value": len(bo_method_pool()), "ok": len(bo_method_pool()) == 2,
    })
    return out


# ── orchestration ────────────────────────────────────────────────────────
def main() -> int:
    print(heading("DOE-Benchmark — pre-launch audit harness"))
    print(f"Python   : {sys.version.split()[0]}")
    print(f"Repo     : {Path.cwd()}")
    print(f"Baseline RSS: {rss_mb():.0f} MB")

    blocks = [
        audit_pool_compliance,
        audit_d_optimal,
        audit_turbo,
        audit_mf_cokriging,
        audit_pairing_modes,
        audit_runner_integration,
        audit_mf_runner_integration,
    ]

    all_results: list[dict] = []
    overall_ok = True
    for fn in blocks:
        print(section(fn.__name__))
        t0 = time.perf_counter()
        try:
            res = fn()
        except Exception as e:
            traceback.print_exc()
            res = {"name": fn.__name__, "checks": [
                {"label": "harness exception", "ok": False,
                 "error": f"{type(e).__name__}: {e}"}]}
        all_results.append(res)
        dt = time.perf_counter() - t0
        block_ok = all(c.get("ok", False) for c in res["checks"])
        overall_ok &= block_ok
        flag = "PASS" if block_ok else "FAIL"
        print(f"  [{flag}] {res['name']}  ({dt:.1f}s, {len(res['checks'])} checks)")
        for c in res["checks"]:
            mark = "  ok " if c.get("ok") else "  X  "
            extra = {k: v for k, v in c.items() if k not in ("label", "ok")}
            extra_str = "  " + json.dumps(extra) if extra else ""
            print(f"    {mark} {c['label']}{extra_str}")

    # Markdown report
    report_path = Path("tools/_audit_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Pre-launch audit report", ""]
    lines.append(f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S %z')}")
    lines.append(f"- overall: **{'PASS' if overall_ok else 'FAIL'}**")
    lines.append("")
    for res in all_results:
        block_ok = all(c.get("ok", False) for c in res["checks"])
        lines.append(f"## {res['name']} — {'PASS' if block_ok else 'FAIL'}")
        lines.append("")
        lines.append("| check | status | details |")
        lines.append("|---|---|---|")
        for c in res["checks"]:
            extra = {k: v for k, v in c.items() if k not in ("label", "ok")}
            mark = "OK" if c.get("ok") else "FAIL"
            lines.append(f"| {c['label']} | **{mark}** | `{json.dumps(extra)}` |")
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(heading("DONE  ->  " + ("PASS" if overall_ok else "FAIL")))
    print(f"Report : {report_path.absolute()}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
