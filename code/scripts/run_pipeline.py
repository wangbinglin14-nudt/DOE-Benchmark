"""
scripts/run_pipeline.py
========================
**One-click DOE-Benchmark pipeline orchestrator** — wires Stage 0 → 10
from ``docs/working-notes/DOE-Benchmark实验设计方案.md`` into a single, resumable, fully
audited command.

Why a Python orchestrator?
--------------------------
The design document (``§Pipeline 总图``) prescribes 11 stages that must
run in a fixed order, with each stage's outputs feeding the next.  Doing
this by hand on Windows ``pwsh`` or Linux ``bash`` is fragile: a single
typo in ``--pool`` or ``--seed-start`` silently invalidates the whole
sweep (see §3.4 危险默认值警告).

This script is the *single* entry-point that:

1. Resolves all stage commands from one declarative table — the names
   never drift from the §0.5 / Stage 8-10 specs.
2. Performs a Stage-1 pre-flight on the configured pool / budgets /
   n_repeats / seed-start before any sweep starts; refuses to launch
   on deviation unless ``--allow-deviation`` is given.
3. Runs each stage as a subprocess, streaming stdout / stderr to the
   console *and* to ``logs/pipeline_<UTC-timestamp>.log`` so failures
   are post-mortem-debuggable.
4. Verifies expected output files after each stage; missing outputs
   are a hard error (not just a warning) so a half-finished sweep
   cannot accidentally trigger Stage 8 figdata generation.
5. Supports ``--profile smoke`` for ~5 min CPU sanity-check runs and
   ``--profile full`` for the publication-grade nature-pool sweep.  A
   non-dry-run full profile requires ``--confirm-full-run``; on Windows
   review workstations the Python entry point itself also refuses selected
   full-profile stages that would launch server-scale experiment work.
6. Supports ``--from-stage`` / ``--to-stage`` / ``--only`` / ``--skip``
   for restartability after manual intervention (e.g. resume from
   Stage 4 once an OOM-killed sweep has been resumed via
   ``--skip-existing``).

Usage
-----

    # Smoke test (~5 min): sanity-check the runner on 1 case.
    python scripts/run_pipeline.py --profile smoke

    # Full publication sweep (server only):
    python scripts/run_pipeline.py --profile full --confirm-full-run

    # Skip the (long) sweep and only re-run analysis + figures + macros
    # (e.g. after editing prepare_figdata.py):
    python scripts/run_pipeline.py --from-stage 3 --confirm-full-run

    # Dry-run: print the resolved commands and verify-paths, do not
    # spawn any subprocess.  Always run this once on a new host to
    # confirm the configuration is what you expect.
    python scripts/run_pipeline.py --dry-run

The default profile mirrors the frozen pre-registration:
``--pool nature --budgets 5 10 20 50 --n-repeats 30 --n-test 1000
--seed-strategy case_repeat --seed-start 1000`` on all 34 cases.
Override any of these via ``--pool`` / ``--budgets`` / ``--n-repeats``
/ ``--cases``.
"""
from __future__ import annotations

import argparse
import dataclasses
import io
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.project_layout import paper_file
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_CONSOLE_OUTPUT_BROKEN = False


def _safe_console_write(text: str) -> None:
    """Best-effort console streaming that never invalidates stage results."""
    global _CONSOLE_OUTPUT_BROKEN
    if _CONSOLE_OUTPUT_BROKEN:
        return
    try:
        sys.stdout.write(text)
        sys.stdout.flush()
    except (BrokenPipeError, OSError, ValueError):
        _CONSOLE_OUTPUT_BROKEN = True

# ── Default knobs (paper/preregistration.md §3, frozen) ──────────────────
PREREG_POOL = "nature"
PREREG_BUDGETS = (5, 10, 20, 50)
PREREG_N_REPEATS = 30
PREREG_N_TEST = 1000
PREREG_SEED_STRATEGY = "case_repeat"
PREREG_SEED_START = 1000

# ── Output directory layout (DOE-Benchmark §0.6 architecture) ────────────
RESULTS_ROOT = ROOT / "benchmark" / "results"
SWEEP_DIR = RESULTS_ROOT / "full_benchmark"
RR_DIR = RESULTS_ROOT / "ranking_reversal"
SC_DIR = RESULTS_ROOT / "sample_complexity_0.20"
RR3_DIR = RESULTS_ROOT / "reliability_recovery"
PAPER_DIR = ROOT / "paper"
FIGDATA_DIR = PAPER_DIR / "figdata"
FIGURES_DIR = PAPER_DIR / "figures"
LOGS_DIR = ROOT / "logs"

# The 34 cases of the frozen pre-registration (DOE-Benchmark §0.6.4 /
# paper/preregistration.md §3).  Hard-coded here so a smoke run that
# downsamples to one case still references the same canonical universe.
ALL_CASES: tuple[str, ...] = (
    "tuned_mass_damper_3d",
    "airfoil_stall_2d", "fin_heat_sink_4d", "multilayer_wall_4d",
    "pressure_vessel_4d", "rocket_isp_4d", "waverider_4d",
    "airfoil_transonic_5d", "cantilever_beam_5d", "casting_solidification_5d",
    "hetero_conduction_2d", "laminar_flame_speed_5d", "membrane_reactor_5d",
    "plate_with_hole_5d", "soil_consolidation_5d", "supersonic_intake_5d",
    "supersonic_nozzle_5d", "thin_plate_buckling_5d", "variable_beam_5d",
    "dam_seepage_6d", "fixed_bed_reactor_6d", "groundwater_contamination_6d",
    "rotor_bemt_6d", "solar_pv_yield_6d", "wind_turbine_power_6d", "wing_vlm_6d",
    "battery_thermal_7d",
    "aircraft_conceptual_8d", "multilayer_wall_8d",
    "oil_reservoir_9d",
    "ten_bar_truss_10d", "variable_beam_10d",
    "multilayer_wall_16d", "variable_beam_20d",
)

# Smoke profile: small case + tiny sweep, used for CI / local sanity.
SMOKE_CASES = ("cantilever_beam_5d",)
SMOKE_BUDGETS = (5, 10)
SMOKE_N_REPEATS = 2
SMOKE_N_TEST = 100
SMOKE_POOL = "default"  # 10-method pool, no BoTorch -> no MemoryError risk
SMOKE_GATE_DIR = RESULTS_ROOT / "_smoke_gate"
LARGE_RUN_CELL_THRESHOLD = 10000


# ─── Stage descriptor ────────────────────────────────────────────────────
@dataclass
class Stage:
    """Declarative description of one pipeline stage.

    Attributes
    ----------
    number : int
        Stage number per DOE-Benchmark §Pipeline 总图 (0..10).
    name : str
        Human-readable label used in banners and logs.
    description : str
        One-line semantic summary.
    build_cmd : Callable
        Function ``(args) -> list[str]`` producing the argv to spawn
        as a subprocess.  Returning ``None`` skips the stage (used by
        Stage 7 ``--with-sota`` opt-in).
    expected_outputs : list[Path]
        Files that must exist after the stage finishes; missing files
        => hard error.  Empty list = no post-condition check.
    fresh_outputs : bool
        If true, expected outputs that existed before the child command
        must be rewritten during the stage.  Use only for derived artifacts
        that should be regenerated every run; resumable sweep CSVs keep this
        false.
    optional : bool
        True for Stage 0 (smoke gate) and Stage 7 (SOTA robustness)
        which are not on the Go-condition critical path.
    """

    number: int
    name: str
    description: str
    build_cmd: Callable[["PipelineConfig"], Optional[List[str]]]
    expected_outputs: List[Path] = field(default_factory=list)
    fresh_outputs: bool = False
    optional: bool = False


# ─── Pipeline configuration ──────────────────────────────────────────────
@dataclass
class PipelineConfig:
    """Resolved knobs after parsing CLI args + applying profile defaults."""

    profile: str
    pool: str
    cases: tuple[str, ...]
    budgets: tuple[int, ...]
    n_repeats: int
    n_test: int
    seed_strategy: str
    seed_start: int
    n_jobs: int
    bo_n_jobs: int
    mem_cap_gb: float
    timeout_s: float
    bo_timeout_s: float
    ref_timeout_s: float
    heartbeat_s: float
    max_wall_clock_h: float
    smoke_dir: Path
    sweep_dir: Path
    rr_dir: Path
    sc_dir: Path
    rr3_dir: Path
    figdata_dir: Path
    figures_dir: Path
    paper_dir: Path
    logs_dir: Path
    allow_deviation: bool
    skip_existing: bool
    with_sota: bool
    with_mfbo: bool
    with_d1_pairings: bool
    python: str
    dry_run: bool

    def resolve_python(self) -> str:
        """Pick the Python interpreter for child processes.

        Order of precedence: ``--python`` flag, ``$PYTHON`` env var,
        ``sys.executable`` of the current process.  We deliberately do
        not fall back to ``"python"`` on PATH: that would silently
        switch interpreters between the orchestrator and its children,
        which is exactly the failure mode the manifest is meant to
        prevent.
        """
        return self.python or os.environ.get("PYTHON") or sys.executable


# ─── Stage command builders ──────────────────────────────────────────────
def _cmd_smoke(cfg: PipelineConfig) -> List[str]:
    """Stage 0: 1-case quick smoke gate.

    Runs a tiny runner probe in an isolated directory.  This intentionally
    does not share ``cfg.sweep_dir``: in the smoke profile Stage 2 performs a
    separate 10-method grid audit, and mixing the two resumable CSVs would let
    ``--skip-existing`` inherit the wrong method grid.
    """
    return [
        cfg.resolve_python(), "-u",
        str(ROOT / "benchmark" / "scripts" / "run_full_benchmark.py"),
        "--cases", *SMOKE_CASES,
        "--pool", "default",
        "--budgets", *(str(b) for b in SMOKE_BUDGETS),
        "--n-repeats", str(SMOKE_N_REPEATS),
        "--n-test", str(SMOKE_N_TEST),
        "--seed-strategy", cfg.seed_strategy,
        "--seed-start", str(cfg.seed_start),
        "--n-jobs", "1",
        "--output-dir", str(cfg.smoke_dir),
        "--quiet",
    ]


def _cmd_validate(cfg: PipelineConfig) -> List[str]:
    """Stage 1: pre-flight ``validate_run_config.py``.

    Checks (i) pool yields the expected method count, (ii) budgets /
    n_repeats / n_test / seed knobs match the frozen pre-registration,
    (iii) the canonical hyper-parameter table agrees with the actual
    pool factories (GAP-9), (iv) all 34 cases are requested.  Exits 1
    on any deviation unless ``--allow-deviation`` is explicit.
    """
    cmd = [
        cfg.resolve_python(),
        str(ROOT / "scripts" / "validate_run_config.py"),
        "--pool", cfg.pool,
        "--budgets", *(str(b) for b in cfg.budgets),
        "--n-repeats", str(cfg.n_repeats),
        "--n-test", str(cfg.n_test),
        "--seed-strategy", cfg.seed_strategy,
        "--seed-start", str(cfg.seed_start),
    ]
    if cfg.cases == ALL_CASES:
        cmd += ["--cases", "ALL"]
    else:
        cmd += ["--cases", *cfg.cases]
    if cfg.allow_deviation or cfg.profile == "smoke":
        cmd += ["--allow-deviation", "--output-dir", str(cfg.sweep_dir)]
    return cmd


def _cmd_dependency_preflight(cfg: PipelineConfig) -> List[str]:
    """Stage 1b: fail fast on broken optional BoTorch/Torch stacks."""
    cmd = [
        cfg.resolve_python(), "-u",
        str(ROOT / "scripts" / "check_optional_dependencies.py"),
        "--pool", cfg.pool,
    ]
    if cfg.profile == "full" and cfg.pool == "nature":
        cmd.append("--require-botorch")
        cmd += ["--expected-methods", "20"]
        cmd.append("--require-formal-case-backends")
        cmd.append("--require-runtime-monitoring")
    return cmd


def _cmd_case_experiment_readiness(cfg: PipelineConfig) -> List[str]:
    """Stage 1a: all-case smoke evaluate audit before long sweeps."""
    return [
        cfg.resolve_python(), "-u",
        str(ROOT / "scripts" / "audit_case_experiment_readiness.py"),
        "--output", str(ROOT / "docs" / "case_experiment_readiness.csv"),
        "--fail-on-warning",
    ]


def _cmd_full_sweep(cfg: PipelineConfig) -> List[str]:
    """Stage 2: 34×20×4×30 = 81 600-cell main sweep.

    On a 16 GB Windows host this takes ~1-2 days even with
    ``--bo-n-jobs 1``; on a 64+ GB Linux server it shrinks to ~30 h.
    The runner is fully resumable (``--skip-existing``) so an OOM kill
    or wall-clock timeout can be restarted by re-invoking this script
    with the same arguments.
    """
    cmd = [
        cfg.resolve_python(), "-u",
        str(ROOT / "benchmark" / "scripts" / "run_full_benchmark.py"),
        "--cases", *cfg.cases,
        "--pool", cfg.pool,
        "--budgets", *(str(b) for b in cfg.budgets),
        "--n-repeats", str(cfg.n_repeats),
        "--n-test", str(cfg.n_test),
        "--seed-strategy", cfg.seed_strategy,
        "--seed-start", str(cfg.seed_start),
        "--n-jobs", str(cfg.n_jobs),
        "--bo-n-jobs", str(cfg.bo_n_jobs),
        "--mem-cap-gb", str(cfg.mem_cap_gb),
        "--timeout-per-cell", str(cfg.timeout_s),
        "--bo-timeout-per-cell", str(cfg.bo_timeout_s),
        "--ref-timeout-s", str(cfg.ref_timeout_s),
        "--heartbeat-s", str(cfg.heartbeat_s),
        "--output-dir", str(cfg.sweep_dir),
    ]
    if cfg.skip_existing:
        cmd.append("--skip-existing")
    if cfg.max_wall_clock_h > 0:
        cmd += ["--max-wall-clock-h", str(cfg.max_wall_clock_h)]
    if _planned_cells(cfg) >= LARGE_RUN_CELL_THRESHOLD:
        cmd.append("--confirm-large-run")
    if cfg.profile == "full" and cfg.pool == "nature":
        cmd.append("--require-botorch")
        cmd.append("--require-formal-case-backends")
        cmd.append("--require-runtime-monitoring")
    return cmd


def _cmd_sweep_audit(cfg: PipelineConfig) -> List[str]:
    """Stage 2b: integrity gate for the streamed ``results.csv``.

    The runner may legitimately return with partial data when a wall-clock
    cap is set or the invocation is used to resume a sweep.  Paper-facing
    stages need a stricter post-condition: the declared case/method/budget
    grid must be present, with at least 28/30 repeats per cell and an
    acceptable success fraction.
    """
    return [
        cfg.resolve_python(), "-u",
        str(ROOT / "scripts" / "audit_sweep_integrity.py"),
        str(cfg.sweep_dir / "results.csv"),
        "--expected-cases", ",".join(cfg.cases),
        "--expected-methods", ",".join(_expected_methods_for_pool(cfg.pool)),
        "--expected-method-families", ",".join(_expected_method_families_for_pool(cfg.pool)),
        "--expected-budgets", *(str(b) for b in cfg.budgets),
        "--expected-repeats", str(cfg.n_repeats),
        "--expected-n-test", str(cfg.n_test),
        "--expected-test-seed", "0",
        "--seed-strategy", cfg.seed_strategy,
        "--seed-start", str(cfg.seed_start),
        "--min-repeats-per-cell", str(min(cfg.n_repeats, 28)),
        "--min-ok-fraction", "0.95",
        "--max-timeout-fraction", "0.05",
        "--run-config-json", str(cfg.sweep_dir / "run_config.json"),
        "--require-run-config",
        "--output", str(cfg.sweep_dir / "sweep_integrity_report.md"),
    ]


def _cmd_headline_report(cfg: PipelineConfig) -> List[str]:
    """Stage 3: post-sweep ``generate_report.py``.

    Produces ``<sweep_dir>/report/REPORT.md`` and
    ``<sweep_dir>/report/figs/`` from
    ``results.csv``.  Per the §Stage 3 Go-condition the report itself
    only emits Markdown + PNGs; downstream CSVs (``stratified_ranks``,
    Pareto, etc.) are recomputed on-demand by ``prepare_figdata.py``.
    """
    return [
        cfg.resolve_python(), "-u",
        str(ROOT / "benchmark" / "scripts" / "generate_report.py"),
        str(cfg.sweep_dir),
    ]


def _cmd_d1(cfg: PipelineConfig) -> List[str]:
    """Stage 4a: ``run_ranking_reversal.py --pairing-mode all``.

    Runs the analytical phase once and the physics phase three times
    (dimension / topology / random pairings, see §4.1).  Emits
    ``correlation.json`` and one ``pairing_<mode>/`` sub-directory each.
    """
    d1_budgets = cfg.budgets if cfg.profile == "smoke" else (10, 20)
    pairing_mode = (
        "dimension"
        if cfg.profile == "smoke"
        else ("all" if cfg.with_d1_pairings else "dimension")
    )
    cmd = [
        cfg.resolve_python(), "-u",
        str(ROOT / "benchmark" / "scripts" / "run_ranking_reversal.py"),
        "--pool", cfg.pool,
        "--n-repeats", str(cfg.n_repeats),
        # D1 only needs the two endpoints that bracket the protocol's
        # primary / secondary budgets (10·d, 20·d).  Adding the 5·d /
        # 50·d endpoints would *double* the analytical cost without
        # changing the headline conclusion.
        "--budgets", *(str(b) for b in d1_budgets),
        "--cases", *(("ALL",) if cfg.cases == ALL_CASES else cfg.cases),
        "--n-test", str(cfg.n_test),
        "--seed-strategy", cfg.seed_strategy,
        "--seed-start", str(cfg.seed_start),
        "--n-jobs", str(cfg.n_jobs),
        "--bo-n-jobs", str(cfg.bo_n_jobs),
        "--mem-cap-gb", str(cfg.mem_cap_gb),
        "--timeout-s", str(cfg.timeout_s),
        "--bo-timeout-s", str(cfg.bo_timeout_s),
        "--ref-timeout-s", str(cfg.ref_timeout_s),
        "--pairing-mode", pairing_mode,
        "--skip-existing",
        "--output-dir", str(cfg.rr_dir),
    ]
    return cmd


def _cmd_d1_ci(cfg: PipelineConfig) -> List[str]:
    """Stage 4b: ``recompute_spearman_ci.py`` — bootstrap 95 % CI on ρ.

    Reads ``correlation.json`` and the (potentially nested
    ``pairing_<mode>/``) ``rank_comparison.csv`` files, augments each
    record with ``spearman_rho_ci_lo/hi`` plus Kendall τ CI, and
    over-writes ``correlation.json`` in place.  GAP-15 fix.
    """
    return [
        cfg.resolve_python(), "-u",
        "-m", "benchmark.scripts.recompute_spearman_ci",
        "--rr-dir", str(cfg.rr_dir),
    ]


def _cmd_d2(cfg: PipelineConfig) -> List[str]:
    """Stage 5: ``run_sample_complexity.py``.

    Reads ``results.csv``, fits per-(method, topology) learning curves,
    and writes ``rule_summary.csv`` containing the 10·d-rule pass-rate
    plus the GAP-11 ``n_cases``/``case_ids``/``exploratory`` columns.
    """
    cmd = [
        cfg.resolve_python(), "-u",
        str(ROOT / "benchmark" / "scripts" / "run_sample_complexity.py"),
        str(cfg.sweep_dir / "results.csv"),
        "--threshold", "0.20",
        "--rule-budget", "10",
        "--output-dir", str(cfg.sc_dir),
    ]
    if cfg.profile == "full" and cfg.pool == "nature":
        cmd.append("--require-headline-source")
    return cmd


def _cmd_d3(cfg: PipelineConfig) -> List[str]:
    """Stage 6: ``run_reliability_recovery.py``.

    Computes per-method signed P_f bias on the 16 cases that declare a
    ``failure_threshold``, with Holm-Bonferroni-corrected p-values
    written by the upstream summariser (DOE-Benchmark §4.3 D3 rule).
    The pre-registered primary interval estimator is the case-cluster
    bootstrap, so this stage must pass ``--cluster-by case_id``.
    """
    cmd = [
        cfg.resolve_python(), "-u",
        str(ROOT / "benchmark" / "scripts" / "run_reliability_recovery.py"),
        str(cfg.sweep_dir / "results.csv"),
        "--budget-mult", "10",
        "--output-aggregate", "max",
        "--cluster-by", "case_id",
        "--output-dir", str(cfg.rr3_dir),
    ]
    if cfg.profile == "full" and cfg.pool == "nature":
        cmd.append("--require-headline-source")
    return cmd


def _d1_expected_outputs(cfg: PipelineConfig) -> List[Path]:
    modes = (
        ("dimension",)
        if cfg.profile == "smoke" or not cfg.with_d1_pairings
        else ("dimension", "topology", "random")
    )
    return [
        cfg.rr_dir / "correlation.json",
        *[
            cfg.rr_dir / f"pairing_{mode}" / "rank_comparison.csv"
            for mode in modes
        ],
    ]


def _cmd_sota(cfg: PipelineConfig) -> Optional[List[str]]:
    """Stage 7a (optional): ``launch_sota_robustness.sh``.

    SOTA robustness amendment §A1: add TURBO + D-optimal to the pool.
    Skipped on Windows (the launcher is a bash script) and unless the
    user explicitly opts in via ``--with-sota``.
    """
    if not cfg.with_sota:
        return None
    if os.name == "nt":
        print("[skip] Stage 7 SOTA robustness: requires bash (Linux/WSL).")
        return None
    return ["bash", str(ROOT / "scripts" / "launch_sota_robustness.sh"), "fg"]


def _cmd_mfbo(cfg: PipelineConfig) -> Optional[List[str]]:
    """Stage 7b (optional): ``launch_mfbo.sh`` — multi-fidelity supplement."""
    if not cfg.with_mfbo:
        return None
    if os.name == "nt":
        print("[skip] Stage 7 MF-BO: requires bash (Linux/WSL).")
        return None
    return ["bash", str(ROOT / "scripts" / "launch_mfbo.sh"), "fg"]


def _cmd_figdata(cfg: PipelineConfig) -> List[str]:
    """Stage 8: ``prepare_figdata.py`` — figdata layer for plots.

    Use the current pipeline configuration as the method contract here.  The
    fresh RUN_MANIFEST is generated later in the pipeline, so passing an
    already-existing manifest at Stage 8 would let a stale archive override the
    current pool and contaminate regenerated figdata.
    """
    cmd = [
        cfg.resolve_python(), "-u",
        str(ROOT / "scripts" / "prepare_figdata.py"),
        "--results-csv", str(cfg.sweep_dir / "results.csv"),
        "--d1-dir", str(cfg.rr_dir),
        "--d2-dir", str(cfg.sc_dir),
        "--d3-dir", str(cfg.rr3_dir),
        "--out-dir", str(cfg.figdata_dir),
        "--expected-methods", ",".join(_expected_methods_for_pool(cfg.pool)),
    ]
    if cfg.profile == "full" and cfg.pool == "nature":
        cmd.append("--require-headline-source")
    return cmd


def _cmd_figures(cfg: PipelineConfig) -> List[str]:
    """Stage 9: ``plots_nature.py`` — Nature-format Fig 1-6."""
    return [
        cfg.resolve_python(), "-u",
        str(ROOT / "scripts" / "plots_nature.py"),
        "--figdata", str(cfg.figdata_dir),
        "--out-dir", str(cfg.figures_dir),
    ]


def _cmd_numbers(cfg: PipelineConfig) -> List[str]:
    """Stage 10a: ``fill_paper_numbers.py`` — paper/numbers.tex macros."""
    cmd = [
        cfg.resolve_python(), "-u",
        str(ROOT / "scripts" / "fill_paper_numbers.py"),
        "--results-csv", str(cfg.sweep_dir / "results.csv"),
        "--rr-dir", str(cfg.rr_dir),
        "--d2-dir", str(cfg.sc_dir),
        "--d3-dir", str(cfg.rr3_dir),
        "--figdata-dir", str(cfg.figdata_dir),
        "--test-protocol-dir", str(cfg.sweep_dir / "analysis"),
        "--output-tex", str(cfg.paper_dir / "numbers.tex"),
        "--output-json", str(cfg.paper_dir / "numbers.json"),
        "--require-results-csv",
        "--require-results-contract",
        "--require-submission-complete",
    ]
    if cfg.profile == "full" and cfg.pool == "nature":
        cmd.append("--require-headline-source")
    return cmd


def _cmd_test_protocol(cfg: PipelineConfig) -> List[str]:
    """Stage 10a.0: test-set MC-noise and metric-robustness analysis."""
    cmd = [
        cfg.resolve_python(), "-u",
        str(ROOT / "benchmark" / "scripts" / "validate_test_protocol.py"),
        str(cfg.sweep_dir),
    ]
    if cfg.profile == "full" and cfg.pool == "nature":
        cmd.append("--require-headline-source")
    return cmd


def _cmd_si(cfg: PipelineConfig) -> List[str]:
    """Stage 10a.5: ``generate_si.py`` — baseline supplementary tables."""
    return [
        cfg.resolve_python(), "-u",
        str(ROOT / "scripts" / "generate_si.py"),
        "--output-dir", str(cfg.paper_dir),
    ]


def _cmd_case_validation_note_sources(cfg: PipelineConfig) -> List[str]:
    """Stage 10a.6: materialized validation-note/reference fingerprints."""
    return [
        cfg.resolve_python(), "-u",
        str(ROOT / "scripts" / "update_case_validation_note_sources.py"),
        "--write",
        "--output", str(ROOT / "docs" / "case_validation_note_sources.csv"),
    ]


def _cmd_case_validation_summary(cfg: PipelineConfig) -> List[str]:
    """Stage 10a.7: materialized case-validation audit summary."""
    return [
        cfg.resolve_python(), "-u",
        str(ROOT / "scripts" / "generate_case_validation_summary.py"),
        "--note-sources-csv", str(ROOT / "docs" / "case_validation_note_sources.csv"),
        "--output", str(ROOT / "docs" / "CASE_VALIDATION_SUMMARY.md"),
        "--strict",
    ]


def _cmd_manifest(cfg: PipelineConfig) -> List[str]:
    """Stage 10b: ``generate_run_manifest.py`` — RUN_MANIFEST.json."""
    reproduction_parts = [
        shlex.quote(cfg.resolve_python()),
        shlex.quote(str(ROOT / "scripts" / "run_pipeline.py")),
        "--profile", shlex.quote(cfg.profile),
    ]
    if cfg.profile == "full":
        reproduction_parts.append("--confirm-full-run")
    reproduction_parts += [
        "--pool", shlex.quote(cfg.pool),
        "--budgets", *(shlex.quote(str(b)) for b in cfg.budgets),
        "--n-repeats", shlex.quote(str(cfg.n_repeats)),
        "--n-test", shlex.quote(str(cfg.n_test)),
        "--seed-strategy", shlex.quote(cfg.seed_strategy),
        "--seed-start", shlex.quote(str(cfg.seed_start)),
        "--sweep-dir", shlex.quote(str(cfg.sweep_dir)),
    ]
    reproduction_command = " ".join(
        reproduction_parts
    )
    cmd = [
        cfg.resolve_python(), "-u",
        str(ROOT / "scripts" / "generate_run_manifest.py"),
        "--results-dir", str(cfg.sweep_dir),
        "--pool", cfg.pool,
        "--budgets", *(str(b) for b in cfg.budgets),
        "--n-repeats", str(cfg.n_repeats),
        "--n-test", str(cfg.n_test),
        "--seed-strategy", cfg.seed_strategy,
        "--seed-start", str(cfg.seed_start),
        "--output", str(cfg.sweep_dir / "RUN_MANIFEST.json"),
        "--command", reproduction_command,
        "--command-file", str(_command_history_path(cfg)),
        "--require-complete-archive",
    ]
    if cfg.cases == ALL_CASES:
        cmd += ["--cases", "ALL"]
    else:
        cmd += ["--cases", *cfg.cases]
    return cmd


def _cmd_submission_ready(cfg: PipelineConfig) -> List[str]:
    """Final submission-readiness gate for paper artifacts and result lineage."""
    return [
        cfg.resolve_python(), "-u",
        str(ROOT / "scripts" / "check_submission_ready.py"),
        str(cfg.paper_dir),
        "--manifest", str(cfg.sweep_dir / "RUN_MANIFEST.json"),
        "--numbers", str(cfg.paper_dir / "numbers.json"),
        "--numbers-tex", str(cfg.paper_dir / "numbers.tex"),
        "--figure-manifest", str(cfg.paper_dir / "figure_manifest.json"),
        "--figdata-dir", str(cfg.figdata_dir),
        "--case-validation-note-sources", str(ROOT / "docs" / "case_validation_note_sources.csv"),
        "--case-validation-summary", str(ROOT / "docs" / "CASE_VALIDATION_SUMMARY.md"),
        "--case-experiment-readiness", str(ROOT / "docs" / "case_experiment_readiness.csv"),
        "--d1-dir", str(cfg.rr_dir),
        "--d2-dir", str(cfg.sc_dir),
        "--d3-dir", str(cfg.rr3_dir),
        "--test-protocol-dir", str(cfg.sweep_dir / "analysis"),
    ]


def _expected_methods_for_pool(pool: str) -> list[str]:
    """Return stable method names for an orchestrator pool.

    Imported lazily so ``run_pipeline.py --help`` remains lightweight.
    """
    from benchmark.methods import (
        bo_method_pool,
        default_method_pool,
        extended_method_pool,
        full_method_pool,
        mf_pool,
        nature_pool,
        sota_pool,
    )

    pools = {
        "default": default_method_pool,
        "extended": extended_method_pool,
        "full": full_method_pool,
        "bo": bo_method_pool,
        "nature": nature_pool,
        "sota": sota_pool,
        "mf": mf_pool,
    }
    if pool not in pools:
        raise ValueError(f"unknown method pool {pool!r}")
    return [m.name for m in pools[pool]()]


def _expected_method_families_for_pool(pool: str) -> list[str]:
    from benchmark import (
        bo_method_pool,
        default_method_pool,
        extended_method_pool,
        full_method_pool,
        nature_pool,
    )
    from benchmark.methods import (
        mf_pool,
        sota_pool,
    )

    pools = {
        "default": default_method_pool,
        "extended": extended_method_pool,
        "full": full_method_pool,
        "bo": bo_method_pool,
        "nature": nature_pool,
        "sota": sota_pool,
        "mf": mf_pool,
    }
    if pool not in pools:
        raise ValueError(f"unknown method pool {pool!r}")
    return [f"{m.name}:{m.family}" for m in pools[pool]()]


def _planned_cells(cfg: PipelineConfig) -> int:
    return len(cfg.cases) * len(_expected_methods_for_pool(cfg.pool)) * len(cfg.budgets) * int(cfg.n_repeats)


def _grid_label(cfg: PipelineConfig) -> str:
    return (
        f"{len(cfg.cases)}×{len(_expected_methods_for_pool(cfg.pool))}"
        f"×{len(cfg.budgets)}×{cfg.n_repeats} "
        f"= {_planned_cells(cfg):,}-cell"
    )


WINDOWS_FORBIDDEN_FULL_STAGES = frozenset({"sweep", "d1"})


def validate_windows_full_profile_boundary(
    cfg: PipelineConfig,
    stages: Sequence[Stage],
) -> list[str]:
    """Reject direct Windows launches of server-scale full-profile experiments."""
    if os.name != "nt" or cfg.profile != "full" or cfg.dry_run:
        return []
    selected = {stage.name for stage in stages}
    forbidden = sorted(selected & WINDOWS_FORBIDDEN_FULL_STAGES)
    if not forbidden:
        return []
    return [
        "Windows review workstations are smoke/preflight/post-return only; "
        f"selected full-profile stage(s) {', '.join(forbidden)} would launch "
        "server-scale experiment work. Use `--profile smoke`, `--dry-run`, "
        "`--to-stage 1`, or run `bash scripts/run_pipeline.sh` on the "
        "designated Linux server."
    ]


# ─── Stage table (the canonical pipeline graph) ──────────────────────────
def build_stages(cfg: PipelineConfig) -> List[Stage]:
    """Return the 0-indexed stage list with config-aware paths.

    The order here is the *only* place that defines the pipeline DAG —
    every CLI knob (``--from-stage``, ``--to-stage``, ``--only``,
    ``--skip``) ultimately filters this list.  Keep it aligned with
    DOE-Benchmark §Pipeline 总图.
    """
    return [
        Stage(
            number=0,
            name="smoke",
            description="1-case default-pool smoke gate (run_full_benchmark.py mini)",
            build_cmd=_cmd_smoke,
            expected_outputs=[cfg.smoke_dir / "results.csv"],
            optional=True,
        ),
        Stage(
            number=1,
            name="validate",
            description="Pre-registration compliance gate",
            build_cmd=_cmd_validate,
            expected_outputs=[],
        ),
        Stage(
            number=1,
            name="case_readiness",
            description="All-case experiment-readiness gate",
            build_cmd=_cmd_case_experiment_readiness,
            expected_outputs=[ROOT / "docs" / "case_experiment_readiness.csv"],
            fresh_outputs=True,
        ),
        Stage(
            number=1,
            name="dependency_preflight",
            description="Optional dependency compatibility gate",
            build_cmd=_cmd_dependency_preflight,
            expected_outputs=[],
        ),
        Stage(
            number=2,
            name="sweep",
            description=f"Main {_grid_label(cfg)} sweep (results.csv)",
            build_cmd=_cmd_full_sweep,
            expected_outputs=[cfg.sweep_dir / "results.csv"],
        ),
        Stage(
            number=2,
            name="sweep_audit",
            description="Integrity gate for full sweep results.csv",
            build_cmd=_cmd_sweep_audit,
            expected_outputs=[cfg.sweep_dir / "sweep_integrity_report.md"],
            fresh_outputs=True,
        ),
        Stage(
            number=3,
            name="report",
            description="Headline REPORT.md + figs/",
            build_cmd=_cmd_headline_report,
            expected_outputs=[
                cfg.sweep_dir / "report" / "REPORT.md",
                cfg.sweep_dir / "report" / "figs",
            ],
            fresh_outputs=True,
        ),
        Stage(
            number=4,
            name="d1",
            description="D1: ranking reversal + Spearman ρ CI",
            build_cmd=_cmd_d1,
            expected_outputs=_d1_expected_outputs(cfg),
        ),
        Stage(
            number=4,
            name="d1_ci",
            description="D1: bootstrap CI augmentation",
            build_cmd=_cmd_d1_ci,
            expected_outputs=_d1_expected_outputs(cfg),
            fresh_outputs=True,
        ),
        Stage(
            number=5,
            name="d2",
            description="D2: per-topology sample complexity",
            build_cmd=_cmd_d2,
            expected_outputs=[
                cfg.sc_dir / "rule_summary.csv",
                cfg.sc_dir / "learning_curves.csv",
                cfg.sc_dir / "n_star_per_topology.csv",
                cfg.sc_dir / "REPORT.md",
            ],
            fresh_outputs=True,
        ),
        Stage(
            number=6,
            name="d3",
            description="D3: signed reliability bias + Holm",
            build_cmd=_cmd_d3,
            expected_outputs=[
                cfg.rr3_dir / "reliability_per_method.csv",
                cfg.rr3_dir / "reliability_per_method_iid.csv",
                cfg.rr3_dir / "reliability_ci_sensitivity.csv",
            ],
            fresh_outputs=True,
        ),
        Stage(
            number=7,
            name="sota",
            description="SOTA robustness (post-reg, optional)",
            build_cmd=_cmd_sota,
            expected_outputs=[],
            optional=True,
        ),
        Stage(
            number=7,
            name="mfbo",
            description="MF-BO supplement (post-reg, optional)",
            build_cmd=_cmd_mfbo,
            expected_outputs=[],
            optional=True,
        ),
        Stage(
            number=8,
            name="figdata",
            description="Figure data package (paper/figdata/*.csv)",
            build_cmd=_cmd_figdata,
            expected_outputs=[
                cfg.figdata_dir / "fig1_case_catalog.csv",
                cfg.figdata_dir / "fig1_method_catalog.csv",
                cfg.figdata_dir / "fig2_d1_ranks.csv",
                cfg.figdata_dir / "fig2_d1_spearman.csv",
                cfg.figdata_dir / "fig3_d2_learning.csv",
                cfg.figdata_dir / "fig3_d2_rule.csv",
                cfg.figdata_dir / "fig4_d3_bias.csv",
                cfg.figdata_dir / "fig5_mechanism.csv",
                cfg.figdata_dir / "fig6_pareto.csv",
                cfg.figdata_dir / "figdata_manifest.json",
            ],
            fresh_outputs=True,
        ),
        Stage(
            number=9,
            name="figures",
            description="Nature-format Fig 1-6 (plots_nature.py)",
            build_cmd=_cmd_figures,
            expected_outputs=[
                cfg.figures_dir / "pdf" / "fig1.pdf",
                cfg.figures_dir / "pdf" / "fig2.pdf",
                cfg.figures_dir / "pdf" / "fig3.pdf",
                cfg.figures_dir / "pdf" / "fig4.pdf",
                cfg.figures_dir / "pdf" / "fig5.pdf",
                cfg.figures_dir / "pdf" / "fig6.pdf",
                cfg.figures_dir / "svg" / "fig1.svg",
                cfg.figures_dir / "svg" / "fig2.svg",
                cfg.figures_dir / "svg" / "fig3.svg",
                cfg.figures_dir / "svg" / "fig4.svg",
                cfg.figures_dir / "svg" / "fig5.svg",
                cfg.figures_dir / "svg" / "fig6.svg",
                cfg.figures_dir / "png" / "fig1.png",
                cfg.figures_dir / "png" / "fig2.png",
                cfg.figures_dir / "png" / "fig3.png",
                cfg.figures_dir / "png" / "fig4.png",
                cfg.figures_dir / "png" / "fig5.png",
                cfg.figures_dir / "png" / "fig6.png",
                cfg.figures_dir / "tiff" / "fig1.tiff",
                cfg.figures_dir / "tiff" / "fig2.tiff",
                cfg.figures_dir / "tiff" / "fig3.tiff",
                cfg.figures_dir / "tiff" / "fig4.tiff",
                cfg.figures_dir / "tiff" / "fig5.tiff",
                cfg.figures_dir / "tiff" / "fig6.tiff",
                cfg.paper_dir / "figure_manifest.json",
            ],
            fresh_outputs=True,
        ),
        Stage(
            number=10,
            name="test_protocol",
            description="Test-set MC-noise and metric-robustness analysis",
            build_cmd=_cmd_test_protocol,
            expected_outputs=[
                cfg.sweep_dir / "analysis" / "test_set_sufficiency.csv",
                cfg.sweep_dir / "analysis" / "test_set_sufficiency.png",
                cfg.sweep_dir / "analysis" / "cross_metric_ranks.csv",
                cfg.sweep_dir / "analysis" / "cross_metric_rank_correlation.csv",
                cfg.sweep_dir / "analysis" / "cross_metric_correlation.png",
                cfg.sweep_dir / "analysis" / "method_robustness.csv",
                cfg.sweep_dir / "analysis" / "method_robustness.png",
            ],
            fresh_outputs=True,
        ),
        Stage(
            number=10,
            name="numbers",
            description="paper/numbers.tex + numbers.json macros",
            build_cmd=_cmd_numbers,
            expected_outputs=[
                cfg.paper_dir / "numbers.tex",
                cfg.paper_dir / "numbers.json",
            ],
            fresh_outputs=True,
        ),
        Stage(
            number=10,
            name="si",
            description="Baseline SI tables (generate_si.py)",
            build_cmd=_cmd_si,
            expected_outputs=[
                paper_file(cfg.paper_dir, "si_hyperparameters.tex"),
                paper_file(cfg.paper_dir, "si_hyperparameters.md"),
                paper_file(cfg.paper_dir, "si_sobol.tex"),
                paper_file(cfg.paper_dir, "si_sobol.md"),
                paper_file(cfg.paper_dir, "si_cases.tex"),
                paper_file(cfg.paper_dir, "si_cases.md"),
                paper_file(cfg.paper_dir, "si_multiple_testing.tex"),
                paper_file(cfg.paper_dir, "si_multiple_testing.md"),
                paper_file(cfg.paper_dir, "si_d2_policy.tex"),
                paper_file(cfg.paper_dir, "si_d2_policy.md"),
            ],
            fresh_outputs=True,
        ),
        Stage(
            number=10,
            name="case_validation_note_sources",
            description="Case validation note/reference fingerprints",
            build_cmd=_cmd_case_validation_note_sources,
            expected_outputs=[ROOT / "docs" / "case_validation_note_sources.csv"],
            fresh_outputs=True,
        ),
        Stage(
            number=10,
            name="case_validation_summary",
            description="Case validation summary (generate_case_validation_summary.py)",
            build_cmd=_cmd_case_validation_summary,
            expected_outputs=[ROOT / "docs" / "CASE_VALIDATION_SUMMARY.md"],
            fresh_outputs=True,
        ),
        Stage(
            number=10,
            name="manifest",
            description="RUN_MANIFEST.json (Stage 10 archive)",
            build_cmd=_cmd_manifest,
            expected_outputs=[cfg.sweep_dir / "RUN_MANIFEST.json"],
            fresh_outputs=True,
        ),
        Stage(
            number=10,
            name="submission_ready",
            description="Final paper artifact + lineage readiness gate",
            build_cmd=_cmd_submission_ready,
            expected_outputs=[],
        ),
    ]


# ─── BLAS thread caps (set before any subprocess) ────────────────────────
def pin_blas_threads() -> None:
    """Force BLAS / OpenMP libraries to single-thread mode.

    Joblib's process-level parallelism oversubscribes the CPU if MKL /
    OpenBLAS / OMP each spawn their own thread pool; on a 16-core box
    that becomes 16 × 16 = 256 threads and the kernel scheduler
    thrashes.  We pin them to 1 so the only parallelism is joblib's
    ``n_jobs``.  This must happen *before* the orchestrator spawns
    children — once a child is forked the env is locked in.  We overwrite
    inherited thread counts intentionally: clusters and module systems often
    export larger values that are unsafe for this process-parallel protocol.
    """
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[var] = "1"
    # Tokenizers: the HF library spams a warning if it's left at default.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ─── Stage selection helpers ─────────────────────────────────────────────
def select_stages(
    stages: Sequence[Stage],
    *,
    from_stage: Optional[int],
    to_stage: Optional[int],
    only: Sequence[str],
    skip: Sequence[str],
    include_smoke: bool,
) -> List[Stage]:
    """Filter the stage list by the user's CLI selection.

    Selection order (every later filter narrows the previous result):

    1. Drop optional Stage 0 unless ``include_smoke`` is True.
    2. Drop everything outside ``[from_stage, to_stage]`` (inclusive).
    3. If ``--only`` is given, keep only stages whose name is listed.
    4. If ``--skip`` is given, drop stages whose name is listed.

    The CLI guarantees a non-empty selection: an empty result raises
    ``ValueError`` so the user gets an immediate, descriptive error
    rather than a silent no-op (which previously masked typos like
    ``--only figdaata``).
    """
    if not include_smoke:
        stages = [s for s in stages if s.name != "smoke"]
    if from_stage is not None:
        stages = [s for s in stages if s.number >= from_stage]
    if to_stage is not None:
        stages = [s for s in stages if s.number <= to_stage]
    if only:
        only_set = set(only)
        unknown = only_set - {s.name for s in stages}
        if unknown:
            raise ValueError(
                f"--only mentions unknown stage names: {sorted(unknown)}; "
                f"available: {sorted({s.name for s in stages})}"
            )
        stages = [s for s in stages if s.name in only_set]
    if skip:
        stages = [s for s in stages if s.name not in set(skip)]
    if not stages:
        raise ValueError(
            "stage selection is empty; check --from-stage/--to-stage/--only/--skip"
        )
    return list(stages)


# ─── Subprocess runner ───────────────────────────────────────────────────
def run_stage(
    stage: Stage,
    cfg: PipelineConfig,
    *,
    log_path: Path,
    continue_on_error: bool,
) -> int:
    """Execute one stage; return its exit code.

    Streams child stdout / stderr line-by-line to:

    * the orchestrator's own stdout (so the user sees live progress);
    * ``log_path`` (so the run is post-mortem-debuggable).

    On success, every path in ``stage.expected_outputs`` is checked;
    a missing path is upgraded to a non-zero return so a stage that
    silently no-op'd cannot trick downstream stages into proceeding.
    For stages marked ``fresh_outputs=True``, outputs that already
    existed before the command must also have a newer modification time.
    The ``continue_on_error`` flag is interpreted by the caller; this
    function still returns a non-zero code for failed postconditions so
    the final pipeline result cannot be falsely green.
    """
    cmd = stage.build_cmd(cfg)
    if cmd is None:
        _safe_console_write(f"\n[skip] Stage {stage.number} {stage.name}: builder declined\n")
        return 0

    banner = (
        f"\n{'=' * 78}\n"
        f"  Stage {stage.number} — {stage.name}\n"
        f"  {stage.description}\n"
        f"  $ {' '.join(shlex.quote(c) for c in cmd)}\n"
        f"{'=' * 78}\n"
    )
    _safe_console_write(banner)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    before_mtime_ns = {
        p: p.stat().st_mtime_ns
        for p in stage.expected_outputs
        if p.is_file()
    }
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(banner)
        logf.flush()
        if cfg.dry_run:
            logf.write("[dry-run] command not executed\n")
            return 0
        t0 = time.perf_counter()
        env = {
            **os.environ,
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
        }
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                bufsize=1,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError as exc:
            msg = f"[error] cannot launch stage {stage.name}: {exc}"
            _safe_console_write(msg + "\n")
            logf.write(msg + "\n")
            return 127
        assert proc.stdout is not None
        for line in proc.stdout:
            _safe_console_write(line)
            logf.write(line)
        rc = proc.wait()
        dt = time.perf_counter() - t0
        summary = f"\n[stage {stage.name}] exit={rc} in {dt:.1f}s\n"
        _safe_console_write(summary + "\n")
        logf.write(summary)
        if rc != 0:
            return rc

    # Post-condition check: missing expected outputs are a hard error
    # even when the child exited cleanly (some stages exit 0 on
    # "nothing to do" paths).  File outputs must also be non-empty, and
    # directory outputs must contain at least one entry; empty artifacts
    # are partial stage results, not publication-ready products.
    missing = [p for p in stage.expected_outputs if not p.exists()]
    if missing:
        msg = (
            f"[error] stage {stage.name} exited 0 but expected outputs are "
            f"missing: {[str(p) for p in missing]}"
        )
        _safe_console_write(msg + "\n")
        with log_path.open("a", encoding="utf-8") as logf:
            logf.write(msg + "\n")
        return 1
    empty_files = [
        p for p in stage.expected_outputs
        if p.is_file() and p.stat().st_size == 0
    ]
    if empty_files:
        msg = (
            f"[error] stage {stage.name} exited 0 but expected output files "
            f"are empty: {[str(p) for p in empty_files]}"
        )
        _safe_console_write(msg + "\n")
        with log_path.open("a", encoding="utf-8") as logf:
            logf.write(msg + "\n")
        return 1
    empty_dirs = [
        p for p in stage.expected_outputs
        if p.is_dir() and not any(p.iterdir())
    ]
    if empty_dirs:
        msg = (
            f"[error] stage {stage.name} exited 0 but expected output "
            f"directories are empty: {[str(p) for p in empty_dirs]}"
        )
        _safe_console_write(msg + "\n")
        with log_path.open("a", encoding="utf-8") as logf:
            logf.write(msg + "\n")
        return 1
    if stage.fresh_outputs:
        stale = [
            p for p in stage.expected_outputs
            if p in before_mtime_ns and p.stat().st_mtime_ns <= before_mtime_ns[p]
        ]
        if stale:
            msg = (
                f"[error] stage {stage.name} exited 0 but expected outputs "
                f"were not refreshed: {[str(p) for p in stale]}"
            )
            _safe_console_write(msg + "\n")
            with log_path.open("a", encoding="utf-8") as logf:
                logf.write(msg + "\n")
            return 1
    return 0


# ─── CLI ─────────────────────────────────────────────────────────────────
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "One-click DOE-Benchmark pipeline orchestrator (Stage 0-10). "
            "See module docstring for the long-form rationale."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g_profile = p.add_argument_group("profile / preset")
    g_profile.add_argument(
        "--profile",
        choices=["smoke", "full"],
        default="full",
        help=(
            "smoke = Stage-0 1-case mini run (~5 min); "
            "full  = current nature-pool publication sweep (~30-50 h on Linux)."
        ),
    )
    g_profile.add_argument(
        "--confirm-full-run",
        action="store_true",
        help=(
            "Required for a non-dry-run full profile.  This prevents an "
            "accidental local launch of the publication-scale sweep; server "
            "launchers should pass it explicitly."
        ),
    )

    g_proto = p.add_argument_group("protocol overrides (rarely needed)")
    g_proto.add_argument("--pool", default=None)
    g_proto.add_argument("--budgets", type=int, nargs="+", default=None)
    g_proto.add_argument("--n-repeats", type=int, default=None)
    g_proto.add_argument("--n-test", type=int, default=None)
    g_proto.add_argument("--seed-strategy", default=None)
    g_proto.add_argument("--seed-start", type=int, default=None)
    g_proto.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Case IDs (default: ALL 34 from the pre-registration).",
    )
    g_proto.add_argument(
        "--allow-deviation",
        action="store_true",
        help="Permit pre-registration deviations; logs to "
             "preregistration_deviations.txt.",
    )

    g_runtime = p.add_argument_group("runtime tuning")
    g_runtime.add_argument("--n-jobs", type=int, default=None)
    g_runtime.add_argument("--bo-n-jobs", type=int, default=None)
    g_runtime.add_argument("--mem-cap-gb", type=float, default=None)
    g_runtime.add_argument("--timeout-s", type=float, default=600.0)
    g_runtime.add_argument("--bo-timeout-s", type=float, default=1800.0)
    g_runtime.add_argument("--ref-timeout-s", type=float, default=600.0)
    g_runtime.add_argument("--heartbeat-s", type=float, default=60.0)
    g_runtime.add_argument(
        "--max-wall-clock-h",
        type=float,
        default=0.0,
        help="0 = no cap; non-zero forwards to run_full_benchmark.py.",
    )
    g_runtime.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Always recompute every cell instead of resuming.",
    )

    g_paths = p.add_argument_group("output paths")
    g_paths.add_argument("--smoke-dir", type=Path, default=None)
    g_paths.add_argument("--sweep-dir", type=Path, default=None)
    g_paths.add_argument("--rr-dir", type=Path, default=None)
    g_paths.add_argument("--sc-dir", type=Path, default=None)
    g_paths.add_argument("--rr3-dir", type=Path, default=None)
    g_paths.add_argument("--figdata-dir", type=Path, default=None)
    g_paths.add_argument("--figures-dir", type=Path, default=None)
    g_paths.add_argument("--paper-dir", type=Path, default=None)
    g_paths.add_argument("--logs-dir", type=Path, default=None)

    g_select = p.add_argument_group("stage selection")
    g_select.add_argument(
        "--with-smoke",
        action="store_true",
        help="Also run Stage 0 smoke gate (skipped by default; smoke "
             "writes to benchmark/results/_smoke).",
    )
    g_select.add_argument(
        "--from-stage", type=int, default=None,
        help="Run only stages with number >= this value (inclusive).",
    )
    g_select.add_argument(
        "--to-stage", type=int, default=None,
        help="Run only stages with number <= this value (inclusive).",
    )
    g_select.add_argument(
        "--only", nargs="+", default=[],
        help="Run only the named stages (e.g. --only d1 d1_ci figdata).",
    )
    g_select.add_argument(
        "--skip", nargs="+", default=[],
        help="Skip the named stages.",
    )
    g_select.add_argument(
        "--with-sota", action="store_true",
        help="Also run Stage 7 SOTA robustness (Linux only).",
    )
    g_select.add_argument(
        "--with-mfbo", action="store_true",
        help="Also run Stage 7 MF-BO supplement (Linux only).",
    )
    g_select.add_argument(
        "--no-d1-pairings", action="store_true",
        help="D1 only runs the dimension pairing (default: all 3).",
    )

    g_misc = p.add_argument_group("misc")
    g_misc.add_argument(
        "--python", default=None,
        help="Python interpreter for child processes (default: $PYTHON or "
             "the one running this script).",
    )
    g_misc.add_argument(
        "--dry-run", action="store_true",
        help="Print resolved commands; do not spawn any subprocess.",
    )
    g_misc.add_argument(
        "--continue-on-error", action="store_true",
        help="Keep going after a failed stage (best-effort).",
    )
    return p.parse_args(argv)


def resolve_config(args: argparse.Namespace) -> PipelineConfig:
    """Merge profile defaults with explicit CLI overrides."""
    if args.profile == "smoke":
        pool = args.pool or SMOKE_POOL
        budgets = tuple(args.budgets) if args.budgets else SMOKE_BUDGETS
        n_repeats = args.n_repeats or SMOKE_N_REPEATS
        n_test = args.n_test or SMOKE_N_TEST
        cases = tuple(args.cases) if args.cases else SMOKE_CASES
        n_jobs = args.n_jobs or 1
        bo_n_jobs = args.bo_n_jobs or 1
        mem_cap = args.mem_cap_gb or 4.0
        default_sweep_dir = RESULTS_ROOT / "_smoke"
        default_smoke_dir = SMOKE_GATE_DIR
        smoke_analysis_base = (args.sweep_dir or default_sweep_dir)
        default_rr_dir = smoke_analysis_base / "ranking_reversal"
        default_sc_dir = smoke_analysis_base / "sample_complexity_0.20"
        default_rr3_dir = smoke_analysis_base / "reliability_recovery"
        default_figdata_dir = smoke_analysis_base / "paper" / "figdata"
        default_figures_dir = smoke_analysis_base / "paper" / "figures"
        default_paper_dir = smoke_analysis_base / "paper"
    else:
        pool = args.pool or PREREG_POOL
        budgets = tuple(args.budgets) if args.budgets else PREREG_BUDGETS
        n_repeats = args.n_repeats or PREREG_N_REPEATS
        n_test = args.n_test or PREREG_N_TEST
        cases = tuple(args.cases) if args.cases else ALL_CASES
        n_jobs = args.n_jobs or (8 if os.name == "nt" else 16)
        bo_n_jobs = args.bo_n_jobs or (1 if os.name == "nt" else 4)
        mem_cap = args.mem_cap_gb or (12.0 if os.name == "nt" else 56.0)
        default_sweep_dir = SWEEP_DIR
        default_smoke_dir = RESULTS_ROOT / "_smoke"
        default_rr_dir = RR_DIR
        default_sc_dir = SC_DIR
        default_rr3_dir = RR3_DIR
        default_figdata_dir = FIGDATA_DIR
        default_figures_dir = FIGURES_DIR
        default_paper_dir = PAPER_DIR
    return PipelineConfig(
        profile=args.profile,
        pool=pool,
        cases=cases,
        budgets=budgets,
        n_repeats=n_repeats,
        n_test=n_test,
        seed_strategy=args.seed_strategy or PREREG_SEED_STRATEGY,
        seed_start=args.seed_start if args.seed_start is not None else PREREG_SEED_START,
        n_jobs=int(n_jobs),
        bo_n_jobs=int(bo_n_jobs),
        mem_cap_gb=float(mem_cap),
        timeout_s=float(args.timeout_s),
        bo_timeout_s=float(args.bo_timeout_s),
        ref_timeout_s=float(args.ref_timeout_s),
        heartbeat_s=float(args.heartbeat_s),
        max_wall_clock_h=float(args.max_wall_clock_h),
        smoke_dir=(args.smoke_dir or default_smoke_dir).resolve(),
        sweep_dir=(args.sweep_dir or default_sweep_dir).resolve(),
        rr_dir=(args.rr_dir or default_rr_dir).resolve(),
        sc_dir=(args.sc_dir or default_sc_dir).resolve(),
        rr3_dir=(args.rr3_dir or default_rr3_dir).resolve(),
        figdata_dir=(args.figdata_dir or default_figdata_dir).resolve(),
        figures_dir=(args.figures_dir or default_figures_dir).resolve(),
        paper_dir=(args.paper_dir or default_paper_dir).resolve(),
        logs_dir=(args.logs_dir or LOGS_DIR).resolve(),
        allow_deviation=bool(args.allow_deviation),
        skip_existing=False if args.profile == "smoke" else not args.no_skip_existing,
        with_sota=bool(args.with_sota),
        with_mfbo=bool(args.with_mfbo),
        with_d1_pairings=not args.no_d1_pairings,
        python=args.python or "",
        dry_run=bool(args.dry_run),
    )


def print_banner(cfg: PipelineConfig, stages: Sequence[Stage]) -> None:
    """Print the pre-flight summary so the operator can sanity-check it."""
    print("=" * 78)
    print(" DOE-Benchmark one-click pipeline")
    print("=" * 78)
    print(f"  profile         : {cfg.profile}")
    print(f"  python          : {cfg.resolve_python()}")
    print(f"  pool            : {cfg.pool}")
    print(f"  budgets         : {list(cfg.budgets)}")
    print(f"  n_repeats       : {cfg.n_repeats}")
    print(f"  n_test          : {cfg.n_test}")
    print(f"  seed strategy   : {cfg.seed_strategy} (start={cfg.seed_start})")
    print(f"  cases           : {len(cfg.cases)}"
          + ("" if cfg.cases == ALL_CASES else f" — {list(cfg.cases)}"))
    print(f"  grid            : {_grid_label(cfg)}")
    print(f"  n_jobs          : {cfg.n_jobs} (light) / {cfg.bo_n_jobs} (heavy)")
    print(f"  mem_cap_gb      : {cfg.mem_cap_gb}")
    print(f"  smoke_dir       : {cfg.smoke_dir}")
    print(f"  sweep_dir       : {cfg.sweep_dir}")
    print(f"  rr_dir          : {cfg.rr_dir}")
    print(f"  sc_dir          : {cfg.sc_dir}")
    print(f"  rr3_dir         : {cfg.rr3_dir}")
    print(f"  figdata_dir     : {cfg.figdata_dir}")
    print(f"  figures_dir     : {cfg.figures_dir}")
    print(f"  paper_dir       : {cfg.paper_dir}")
    print(f"  dry_run         : {cfg.dry_run}")
    print(f"  allow_deviation : {cfg.allow_deviation}")
    print(f"  skip_existing   : {cfg.skip_existing}")
    print()
    print("  Stages selected:")
    for s in stages:
        flag = " (optional)" if s.optional else ""
        print(f"    [{s.number}] {s.name:<10s} — {s.description}{flag}")
    print("=" * 78, flush=True)


def _command_history_path(cfg: PipelineConfig) -> Path:
    suffix = ".dry_run" if cfg.dry_run else ""
    return cfg.logs_dir / f"pipeline_command_history{suffix}.txt"


def _write_command_history(
    cfg: PipelineConfig,
    stages: Sequence[Stage],
    invocation_argv: Sequence[str],
) -> Path:
    """Materialize the exact stage commands resolved for this pipeline run."""
    path = _command_history_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    invocation = " ".join(
        shlex.quote(part)
        for part in [cfg.resolve_python(), str(ROOT / "scripts" / "run_pipeline.py"), *invocation_argv]
    )
    lines = [
        "# DOE-Benchmark resolved pipeline stage commands",
        "# Format: stage_name<TAB>shell_command",
        f"pipeline_invocation\t{invocation}",
    ]
    for stage in stages:
        cmd = stage.build_cmd(cfg)
        if cmd is None:
            continue
        shell_cmd = " ".join(shlex.quote(part) for part in cmd)
        lines.append(f"{stage.name}\t{shell_cmd}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Execute the end-to-end analysis pipeline (D1/D2/D3, reports, figure data, paper numbers) in dependency order with fail-fast gating."""
    args = parse_args(argv)
    pin_blas_threads()
    cfg = resolve_config(args)
    if cfg.profile == "full" and not cfg.dry_run and not args.confirm_full_run:
        print(
            "[error] refusing to start the full publication pipeline without "
            "--confirm-full-run. Use --profile smoke for local checks, "
            "--dry-run to inspect commands, or run the confirmed full profile "
            "on the designated server.",
            file=sys.stderr,
        )
        return 2
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    cfg.smoke_dir.mkdir(parents=True, exist_ok=True)
    cfg.sweep_dir.mkdir(parents=True, exist_ok=True)

    all_stages = build_stages(cfg)
    smoke_pipeline_default = (
        args.profile == "smoke"
        and not args.with_smoke
        and args.from_stage is None
        and args.to_stage is None
        and not args.only
        and not args.skip
    )
    include_smoke = bool(args.with_smoke or args.profile == "smoke")
    to_stage = 3 if smoke_pipeline_default else args.to_stage
    try:
        stages = select_stages(
            all_stages,
            from_stage=args.from_stage,
            to_stage=to_stage,
            only=args.only,
            skip=args.skip,
            include_smoke=include_smoke,
        )
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    boundary_errors = validate_windows_full_profile_boundary(cfg, stages)
    if boundary_errors:
        for err in boundary_errors:
            print(f"[error] {err}", file=sys.stderr)
        return 2

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    log_path = cfg.logs_dir / f"pipeline_{timestamp}.log"
    invocation_argv = list(argv) if argv is not None else sys.argv[1:]
    command_history_path = _write_command_history(cfg, stages, invocation_argv)
    print_banner(cfg, stages)
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(
            f"# DOE-Benchmark pipeline run @ {timestamp}\n"
            f"# python: {cfg.resolve_python()}\n"
            f"# argv:   {sys.argv}\n"
            f"# config: {dataclasses.asdict(cfg)}\n"
            f"# command_history: {command_history_path}\n\n"
        )

    overall_t0 = time.perf_counter()
    failures: list[str] = []
    for stage in stages:
        rc = run_stage(
            stage, cfg,
            log_path=log_path,
            continue_on_error=bool(args.continue_on_error),
        )
        if rc != 0:
            failures.append(f"{stage.name} (rc={rc})")
            if not args.continue_on_error:
                print(
                    f"\n[abort] stage '{stage.name}' failed with rc={rc}; "
                    f"see {log_path}",
                    file=sys.stderr,
                )
                return rc

    elapsed = time.perf_counter() - overall_t0
    print()
    print("=" * 78)
    if failures:
        print(f"  Pipeline finished with {len(failures)} failed stage(s):")
        for f in failures:
            print(f"    - {f}")
        print(f"  log: {log_path}")
        print("=" * 78)
        return 1
    print(f"  Pipeline completed in {elapsed/60:.1f} min")
    print(f"  log: {log_path}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
