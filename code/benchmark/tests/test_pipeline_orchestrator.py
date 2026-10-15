"""Tests for ``scripts/run_pipeline.py`` — the one-click orchestrator.

These tests pin the *contracts* that downstream operators rely on:

* The pipeline DAG is built deterministically from one declarative table
  in ``build_stages``; no stage names change between releases.
* CLI selection (``--from-stage`` / ``--to-stage`` / ``--only`` / ``--skip``)
  is total: a typo in ``--only`` is a hard error, not a silent no-op.
* ``--profile smoke`` resolves to a 1-case mini sweep; ``--profile full``
  resolves to the frozen pre-registration knobs.
* ``--dry-run`` prints commands without spawning subprocesses.
* The Stage 1 validate gate command always carries ``--allow-deviation``
  iff the user asked for it.

We deliberately do NOT exercise ``run_stage`` against real subprocesses
here — that's the integration tests' job.  These tests should run in
under a second on any laptop.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def run_pipeline():
    """Import ``scripts/run_pipeline.py`` as a module without polluting sys.path.

    We use ``importlib.util.spec_from_file_location`` so the test file
    can sit alongside the existing benchmark tests yet still pull in the
    orchestrator that lives under ``scripts/``.
    """
    path = ROOT / "scripts" / "run_pipeline.py"
    spec = importlib.util.spec_from_file_location("run_pipeline", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_pipeline"] = mod
    spec.loader.exec_module(mod)
    return mod


# ─── Stage table integrity ───────────────────────────────────────────────
class TestStageTable:
    """The stage list is the single source of truth for the pipeline DAG.

    Renaming a stage or changing its number requires updating the
    paper, the README, and the wrapper scripts.  We pin the names and
    numbers here so accidental drift fails CI.
    """

    def test_expected_stage_names(self, run_pipeline):
        cfg = self._cfg(run_pipeline, profile="full")
        names = [s.name for s in run_pipeline.build_stages(cfg)]
        assert names == [
            "smoke", "validate", "case_readiness", "dependency_preflight", "sweep", "sweep_audit", "report",
            "d1", "d1_ci", "d2", "d3",
            "sota", "mfbo",
            "figdata", "figures", "test_protocol", "numbers", "si", "case_validation_note_sources",
            "case_validation_summary",
            "manifest", "submission_ready",
        ]

    def test_stage_numbers_match_design(self, run_pipeline):
        cfg = self._cfg(run_pipeline, profile="full")
        expected = {
            "smoke": 0, "validate": 1, "case_readiness": 1, "dependency_preflight": 1,
            "sweep": 2, "sweep_audit": 2, "report": 3,
            "d1": 4, "d1_ci": 4, "d2": 5, "d3": 6,
            "sota": 7, "mfbo": 7,
            "figdata": 8, "figures": 9,
            "test_protocol": 10, "numbers": 10, "si": 10, "case_validation_note_sources": 10,
            "case_validation_summary": 10,
            "manifest": 10, "submission_ready": 10,
        }
        for s in run_pipeline.build_stages(cfg):
            assert s.number == expected[s.name], (
                f"Stage {s.name} numbered {s.number}, expected {expected[s.name]}"
            )

    def test_optional_flags(self, run_pipeline):
        cfg = self._cfg(run_pipeline, profile="full")
        optional = {s.name for s in run_pipeline.build_stages(cfg) if s.optional}
        # smoke is skipped by default in the full profile; sota/mfbo are
        # post-registration extensions.
        assert optional == {"smoke", "sota", "mfbo"}

    def test_manifest_freeze_follows_fresh_sweep_audit(self, run_pipeline):
        cfg = self._cfg(run_pipeline, profile="full")
        stages = run_pipeline.build_stages(cfg)
        positions = {s.name: i for i, s in enumerate(stages)}

        assert positions["sweep_audit"] < positions["manifest"] < positions["submission_ready"]
        sweep_audit = stages[positions["sweep_audit"]]
        assert cfg.sweep_dir / "sweep_integrity_report.md" in sweep_audit.expected_outputs
        assert sweep_audit.fresh_outputs is True

    def test_sweep_description_uses_current_pool_cardinality(self, run_pipeline):
        cfg = self._cfg(run_pipeline, profile="full")
        n_methods = len(run_pipeline._expected_methods_for_pool(cfg.pool))
        expected_cells = len(cfg.cases) * n_methods * len(cfg.budgets) * cfg.n_repeats

        stage = next(s for s in run_pipeline.build_stages(cfg) if s.name == "sweep")

        assert f"34×{n_methods}×4×30" in stage.description
        assert f"{expected_cells:,}-cell" in stage.description

    @staticmethod
    def _cfg(rp, *, profile):
        args = rp.parse_args(["--profile", profile, "--dry-run"])
        return rp.resolve_config(args)


# ─── Profile defaults ────────────────────────────────────────────────────
class TestProfiles:
    def test_smoke_uses_one_case_default_pool(self, run_pipeline):
        args = run_pipeline.parse_args(["--profile", "smoke", "--dry-run"])
        cfg = run_pipeline.resolve_config(args)
        assert cfg.profile == "smoke"
        assert cfg.pool == run_pipeline.SMOKE_POOL
        assert cfg.cases == run_pipeline.SMOKE_CASES
        assert cfg.budgets == run_pipeline.SMOKE_BUDGETS
        assert cfg.n_repeats == run_pipeline.SMOKE_N_REPEATS
        assert cfg.n_test == run_pipeline.SMOKE_N_TEST
        assert cfg.smoke_dir != cfg.sweep_dir
        assert cfg.smoke_dir.name == "_smoke_gate"
        assert cfg.sweep_dir.name == "_smoke"
        assert cfg.skip_existing is False

    def test_full_matches_prereg(self, run_pipeline):
        args = run_pipeline.parse_args(["--profile", "full", "--dry-run"])
        cfg = run_pipeline.resolve_config(args)
        assert cfg.pool == run_pipeline.PREREG_POOL
        assert cfg.budgets == run_pipeline.PREREG_BUDGETS
        assert cfg.n_repeats == run_pipeline.PREREG_N_REPEATS
        assert cfg.n_test == run_pipeline.PREREG_N_TEST
        assert cfg.seed_strategy == run_pipeline.PREREG_SEED_STRATEGY
        assert cfg.seed_start == run_pipeline.PREREG_SEED_START
        # All 34 cases of the frozen pre-reg.
        assert cfg.cases == run_pipeline.ALL_CASES
        assert len(cfg.cases) == 34
        assert cfg.smoke_dir != cfg.sweep_dir
        assert cfg.smoke_dir.name == "_smoke"
        assert cfg.skip_existing is True

    def test_explicit_overrides_win(self, run_pipeline):
        argv = [
            "--profile", "full",
            "--pool", "default",
            "--budgets", "5", "10",
            "--n-repeats", "5",
            "--cases", "cantilever_beam_5d",
            "--dry-run",
        ]
        cfg = run_pipeline.resolve_config(run_pipeline.parse_args(argv))
        assert cfg.pool == "default"
        assert cfg.budgets == (5, 10)
        assert cfg.n_repeats == 5
        assert cfg.cases == ("cantilever_beam_5d",)


# ─── Stage selection ─────────────────────────────────────────────────────
class TestSelectStages:
    """``select_stages`` is the sole filter behind every CLI knob.

    We test each filter in isolation so a regression points at the
    exact predicate that broke.
    """

    def test_default_skips_smoke(self, run_pipeline):
        cfg = self._cfg(run_pipeline)
        all_stages = run_pipeline.build_stages(cfg)
        chosen = run_pipeline.select_stages(
            all_stages, from_stage=None, to_stage=None,
            only=[], skip=[], include_smoke=False,
        )
        assert "smoke" not in [s.name for s in chosen]

    def test_with_smoke_includes_stage_zero(self, run_pipeline):
        cfg = self._cfg(run_pipeline)
        all_stages = run_pipeline.build_stages(cfg)
        chosen = run_pipeline.select_stages(
            all_stages, from_stage=None, to_stage=None,
            only=[], skip=[], include_smoke=True,
        )
        assert chosen[0].name == "smoke"

    def test_from_stage_inclusive(self, run_pipeline):
        cfg = self._cfg(run_pipeline)
        all_stages = run_pipeline.build_stages(cfg)
        chosen = run_pipeline.select_stages(
            all_stages, from_stage=4, to_stage=None,
            only=[], skip=[], include_smoke=False,
        )
        assert all(s.number >= 4 for s in chosen)
        # Must still include both Stage 4 entries (d1 and d1_ci).
        assert {"d1", "d1_ci"}.issubset({s.name for s in chosen})

    def test_to_stage_inclusive(self, run_pipeline):
        cfg = self._cfg(run_pipeline)
        all_stages = run_pipeline.build_stages(cfg)
        chosen = run_pipeline.select_stages(
            all_stages, from_stage=None, to_stage=2,
            only=[], skip=[], include_smoke=False,
        )
        assert all(s.number <= 2 for s in chosen)
        assert {s.name for s in chosen} == {
            "validate", "case_readiness", "dependency_preflight", "sweep", "sweep_audit",
        }

    def test_only_filters_to_named(self, run_pipeline):
        cfg = self._cfg(run_pipeline)
        all_stages = run_pipeline.build_stages(cfg)
        chosen = run_pipeline.select_stages(
            all_stages, from_stage=None, to_stage=None,
            only=["figdata", "figures"], skip=[], include_smoke=False,
        )
        assert [s.name for s in chosen] == ["figdata", "figures"]

    def test_skip_drops_named(self, run_pipeline):
        cfg = self._cfg(run_pipeline)
        all_stages = run_pipeline.build_stages(cfg)
        chosen = run_pipeline.select_stages(
            all_stages, from_stage=None, to_stage=None,
            only=[], skip=["sweep"], include_smoke=False,
        )
        assert "sweep" not in [s.name for s in chosen]

    def test_only_with_typo_raises(self, run_pipeline):
        """A typo in --only must surface as an explicit error."""
        cfg = self._cfg(run_pipeline)
        all_stages = run_pipeline.build_stages(cfg)
        with pytest.raises(ValueError, match="unknown stage names"):
            run_pipeline.select_stages(
                all_stages, from_stage=None, to_stage=None,
                only=["figdaata"], skip=[], include_smoke=False,
            )

    def test_empty_selection_raises(self, run_pipeline):
        """Conflicting filters that select nothing raise so the caller notices."""
        cfg = self._cfg(run_pipeline)
        all_stages = run_pipeline.build_stages(cfg)
        with pytest.raises(ValueError, match="empty"):
            run_pipeline.select_stages(
                all_stages, from_stage=10, to_stage=2,
                only=[], skip=[], include_smoke=False,
            )

    @staticmethod
    def _cfg(rp):
        return rp.resolve_config(rp.parse_args(["--profile", "full", "--dry-run"]))


# ─── Stage command builders ──────────────────────────────────────────────
class TestStageCommands:
    """Spot-check the argv each stage emits.

    Each test pins the smallest invariant that downstream tooling relies
    on; we don't try to cover every flag combination.
    """

    def test_validate_carries_pool_and_budgets(self, run_pipeline):
        cfg = run_pipeline.resolve_config(
            run_pipeline.parse_args(["--profile", "full", "--dry-run"])
        )
        cmd = run_pipeline._cmd_validate(cfg)
        assert "validate_run_config.py" in " ".join(cmd)
        assert "--pool" in cmd and "nature" in cmd
        assert "--budgets" in cmd
        for b in (5, 10, 20, 50):
            assert str(b) in cmd
        # Default does NOT pass --allow-deviation.
        assert "--allow-deviation" not in cmd

    def test_validate_with_allow_deviation(self, run_pipeline):
        cfg = run_pipeline.resolve_config(run_pipeline.parse_args([
            "--profile", "full", "--allow-deviation", "--dry-run",
        ]))
        cmd = run_pipeline._cmd_validate(cfg)
        assert "--allow-deviation" in cmd

    def test_smoke_validate_records_expected_deviation(self, run_pipeline):
        cfg = run_pipeline.resolve_config(run_pipeline.parse_args([
            "--profile", "smoke", "--dry-run",
        ]))
        cmd = run_pipeline._cmd_validate(cfg)

        assert "--allow-deviation" in cmd
        assert "--output-dir" in cmd
        assert cmd[cmd.index("--output-dir") + 1] == str(cfg.sweep_dir)

    def test_smoke_uses_default_pool(self, run_pipeline):
        cfg = run_pipeline.resolve_config(
            run_pipeline.parse_args(["--profile", "smoke", "--dry-run"])
        )
        cmd = run_pipeline._cmd_smoke(cfg)
        # The smoke gate is NOT supposed to use the nature pool — it must
        # finish in minutes on a laptop, so it always falls back to the
        # 10-method default pool.
        assert "default" in cmd
        assert "nature" not in cmd
        assert cmd[cmd.index("--n-test") + 1] == str(run_pipeline.SMOKE_N_TEST)

    def test_full_with_smoke_keeps_smoke_out_of_publication_sweep_dir(self, run_pipeline, tmp_path):
        cfg = run_pipeline.resolve_config(
            run_pipeline.parse_args([
                "--profile", "full",
                "--with-smoke",
                "--sweep-dir", str(tmp_path / "full_benchmark"),
                "--dry-run",
            ])
        )
        cmd = run_pipeline._cmd_smoke(cfg)

        assert cfg.sweep_dir == tmp_path / "full_benchmark"
        assert cfg.smoke_dir != cfg.sweep_dir
        assert cmd[cmd.index("--output-dir") + 1] == str(cfg.smoke_dir)
        assert cmd[cmd.index("--output-dir") + 1] != str(cfg.sweep_dir)

    def test_smoke_profile_sweep_dir_does_not_alias_smoke_gate_dir(self, run_pipeline, tmp_path):
        cfg = run_pipeline.resolve_config(
            run_pipeline.parse_args([
                "--profile", "smoke",
                "--sweep-dir", str(tmp_path / "custom_smoke"),
                "--dry-run",
            ])
        )
        cmd = run_pipeline._cmd_smoke(cfg)

        assert cfg.smoke_dir != tmp_path / "custom_smoke"
        assert cfg.sweep_dir == tmp_path / "custom_smoke"
        assert cmd[cmd.index("--output-dir") + 1] == str(cfg.smoke_dir)

    def test_smoke_profile_stage_zero_uses_isolated_gate_dir(self, run_pipeline, tmp_path):
        cfg = run_pipeline.resolve_config(
            run_pipeline.parse_args([
                "--profile", "smoke",
                "--dry-run",
            ])
        )
        cmd_smoke = run_pipeline._cmd_smoke(cfg)
        cmd_sweep = run_pipeline._cmd_full_sweep(cfg)

        assert cfg.smoke_dir != cfg.sweep_dir
        assert cmd_smoke[cmd_smoke.index("--output-dir") + 1] == str(cfg.smoke_dir)
        assert cmd_sweep[cmd_sweep.index("--output-dir") + 1] == str(cfg.sweep_dir)
        assert cmd_smoke[cmd_smoke.index("--output-dir") + 1] != cmd_sweep[
            cmd_sweep.index("--output-dir") + 1
        ]
        assert "--confirm-large-run" not in cmd_smoke
        assert "--confirm-large-run" not in cmd_sweep
        assert "--skip-existing" not in cmd_sweep

    def test_full_profile_sweep_confirms_large_runner_grid(self, run_pipeline):
        cfg = run_pipeline.resolve_config(
            run_pipeline.parse_args(["--profile", "full", "--dry-run"])
        )
        cmd = run_pipeline._cmd_full_sweep(cfg)

        assert "--confirm-large-run" in cmd
        assert "--require-botorch" in cmd
        assert "--require-formal-case-backends" in cmd
        assert "--require-runtime-monitoring" in cmd

    def test_smoke_profile_analysis_outputs_are_sweep_scoped(self, run_pipeline, tmp_path):
        cfg = run_pipeline.resolve_config(
            run_pipeline.parse_args([
                "--profile", "smoke",
                "--sweep-dir", str(tmp_path / "smoke_run"),
                "--dry-run",
            ])
        )

        assert cfg.rr_dir == cfg.sweep_dir / "ranking_reversal"
        assert cfg.sc_dir == cfg.sweep_dir / "sample_complexity_0.20"
        assert cfg.rr3_dir == cfg.sweep_dir / "reliability_recovery"
        assert cfg.figdata_dir == cfg.sweep_dir / "paper" / "figdata"
        assert cfg.figures_dir == cfg.sweep_dir / "paper" / "figures"
        assert cfg.paper_dir == cfg.sweep_dir / "paper"
        assert cfg.paper_dir != run_pipeline.PAPER_DIR

    def test_dependency_preflight_carries_pool(self, run_pipeline):
        cfg = run_pipeline.resolve_config(
            run_pipeline.parse_args(["--profile", "full", "--dry-run"])
        )
        cmd = run_pipeline._cmd_dependency_preflight(cfg)
        assert "check_optional_dependencies.py" in " ".join(cmd)
        assert "--pool" in cmd
        assert cmd[cmd.index("--pool") + 1] == "nature"
        assert "--require-botorch" in cmd
        assert cmd[cmd.index("--expected-methods") + 1] == "20"
        assert "--require-formal-case-backends" in cmd
        assert "--require-runtime-monitoring" in cmd

    def test_smoke_dependency_preflight_does_not_require_botorch(self, run_pipeline):
        cfg = run_pipeline.resolve_config(
            run_pipeline.parse_args(["--profile", "smoke", "--dry-run"])
        )
        cmd = run_pipeline._cmd_dependency_preflight(cfg)
        assert cmd[cmd.index("--pool") + 1] == "default"
        assert "--require-botorch" not in cmd
        assert "--require-formal-case-backends" not in cmd
        assert "--require-runtime-monitoring" not in cmd

    def test_case_readiness_stage_writes_all_case_audit_csv(self, run_pipeline):
        cfg = run_pipeline.resolve_config(
            run_pipeline.parse_args(["--profile", "full", "--dry-run"])
        )
        cmd = run_pipeline._cmd_case_experiment_readiness(cfg)
        assert "audit_case_experiment_readiness.py" in " ".join(cmd)
        assert "--output" in cmd
        assert "--fail-on-warning" in cmd
        assert "case_experiment_readiness.csv" in cmd[cmd.index("--output") + 1]

        stage = next(s for s in run_pipeline.build_stages(cfg) if s.name == "case_readiness")
        assert stage.expected_outputs == [
            run_pipeline.ROOT / "docs" / "case_experiment_readiness.csv"
        ]
        assert stage.fresh_outputs is True

    def test_d1_uses_two_endpoint_budgets(self, run_pipeline):
        cfg = run_pipeline.resolve_config(
            run_pipeline.parse_args(["--profile", "full", "--dry-run"])
        )
        cmd = run_pipeline._cmd_d1(cfg)
        # D1 is per the protocol restricted to 10·d / 20·d (§4.1) even
        # if the main sweep also runs 5·d / 50·d.
        idx = cmd.index("--budgets")
        assert cmd[idx + 1: idx + 3] == ["10", "20"]

    def test_d1_full_profile_uses_all_physics_cases(self, run_pipeline):
        cfg = run_pipeline.resolve_config(
            run_pipeline.parse_args(["--profile", "full", "--dry-run"])
        )
        cmd = run_pipeline._cmd_d1(cfg)

        idx = cmd.index("--cases")
        assert cmd[idx + 1] == "ALL"

    def test_d1_smoke_profile_stays_smoke_scoped(self, run_pipeline):
        cfg = run_pipeline.resolve_config(
            run_pipeline.parse_args(["--profile", "smoke", "--dry-run"])
        )
        cmd = run_pipeline._cmd_d1(cfg)

        b_idx = cmd.index("--budgets")
        c_idx = cmd.index("--cases")
        p_idx = cmd.index("--pairing-mode")
        assert cmd[b_idx + 1: b_idx + 3] == [str(b) for b in run_pipeline.SMOKE_BUDGETS]
        assert cmd[c_idx + 1: c_idx + 1 + len(run_pipeline.SMOKE_CASES)] == list(run_pipeline.SMOKE_CASES)
        assert cmd[p_idx + 1] == "dimension"

    def test_d1_pairings_default_all(self, run_pipeline):
        cfg = run_pipeline.resolve_config(
            run_pipeline.parse_args(["--profile", "full", "--dry-run"])
        )
        cmd = run_pipeline._cmd_d1(cfg)
        assert "--pairing-mode" in cmd
        idx = cmd.index("--pairing-mode")
        assert cmd[idx + 1] == "all"

    def test_d1_pairings_dimension_only(self, run_pipeline):
        cfg = run_pipeline.resolve_config(run_pipeline.parse_args([
            "--profile", "full", "--no-d1-pairings", "--dry-run",
        ]))
        cmd = run_pipeline._cmd_d1(cfg)
        idx = cmd.index("--pairing-mode")
        assert cmd[idx + 1] == "dimension"

    def test_d1_expected_outputs_cover_selected_pairing_layout(self, run_pipeline):
        cfg = run_pipeline.resolve_config(
            run_pipeline.parse_args(["--profile", "full", "--dry-run"])
        )
        stage = next(s for s in run_pipeline.build_stages(cfg) if s.name == "d1")
        rels = {
            p.relative_to(cfg.rr_dir).as_posix()
            for p in stage.expected_outputs
        }

        assert rels == {
            "correlation.json",
            "pairing_dimension/rank_comparison.csv",
            "pairing_topology/rank_comparison.csv",
            "pairing_random/rank_comparison.csv",
        }

    def test_d1_expected_outputs_respect_dimension_only_mode(self, run_pipeline):
        cfg = run_pipeline.resolve_config(run_pipeline.parse_args([
            "--profile", "full", "--no-d1-pairings", "--dry-run",
        ]))
        stage = next(s for s in run_pipeline.build_stages(cfg) if s.name == "d1")
        rels = {
            p.relative_to(cfg.rr_dir).as_posix()
            for p in stage.expected_outputs
        }

        assert rels == {
            "correlation.json",
            "pairing_dimension/rank_comparison.csv",
        }

    def test_d2_expected_outputs_include_report_and_n_star(self, run_pipeline):
        cfg = run_pipeline.resolve_config(
            run_pipeline.parse_args(["--profile", "full", "--dry-run"])
        )
        cmd = run_pipeline._cmd_d2(cfg)
        assert "run_sample_complexity.py" in " ".join(cmd)
        assert "--require-headline-source" in cmd

        stage = next(s for s in run_pipeline.build_stages(cfg) if s.name == "d2")
        names = {p.name for p in stage.expected_outputs}

        assert names == {
            "rule_summary.csv",
            "learning_curves.csv",
            "n_star_per_topology.csv",
            "REPORT.md",
        }

    def test_d3_uses_preregistered_cluster_bootstrap(self, run_pipeline):
        cfg = run_pipeline.resolve_config(
            run_pipeline.parse_args(["--profile", "full", "--dry-run"])
        )
        cmd = run_pipeline._cmd_d3(cfg)
        assert "run_reliability_recovery.py" in " ".join(cmd)
        assert "--output-aggregate" in cmd
        agg_idx = cmd.index("--output-aggregate")
        assert cmd[agg_idx + 1] == "max"
        assert "--cluster-by" in cmd
        cluster_idx = cmd.index("--cluster-by")
        assert cmd[cluster_idx + 1] == "case_id"
        assert "--require-headline-source" in cmd

    def test_d3_expected_outputs_include_iid_sensitivity(self, run_pipeline):
        cfg = run_pipeline.resolve_config(
            run_pipeline.parse_args(["--profile", "full", "--dry-run"])
        )
        stage = next(s for s in run_pipeline.build_stages(cfg) if s.name == "d3")
        names = {p.name for p in stage.expected_outputs}
        assert "reliability_per_method.csv" in names
        assert "reliability_per_method_iid.csv" in names
        assert "reliability_ci_sensitivity.csv" in names

    def test_report_expected_outputs_match_generate_report_layout(self, run_pipeline):
        cfg = run_pipeline.resolve_config(
            run_pipeline.parse_args(["--profile", "full", "--dry-run"])
        )
        stage = next(s for s in run_pipeline.build_stages(cfg) if s.name == "report")

        assert cfg.sweep_dir / "report" / "REPORT.md" in stage.expected_outputs
        assert cfg.sweep_dir / "report" / "figs" in stage.expected_outputs
        assert cfg.sweep_dir / "REPORT.md" not in stage.expected_outputs

    def test_figdata_expected_outputs_cover_all_main_figure_csvs(self, run_pipeline):
        cfg = run_pipeline.resolve_config(
            run_pipeline.parse_args(["--profile", "full", "--dry-run"])
        )
        stage = next(s for s in run_pipeline.build_stages(cfg) if s.name == "figdata")
        names = {p.name for p in stage.expected_outputs}

        assert names == {
            "fig1_case_catalog.csv",
            "fig1_method_catalog.csv",
            "fig2_d1_ranks.csv",
            "fig2_d1_spearman.csv",
            "fig3_d2_learning.csv",
            "fig3_d2_rule.csv",
            "fig4_d3_bias.csv",
            "fig5_mechanism.csv",
            "fig6_pareto.csv",
            "figdata_manifest.json",
        }

    def test_figures_expected_outputs_cover_all_four_export_formats(self, run_pipeline):
        cfg = run_pipeline.resolve_config(
            run_pipeline.parse_args(["--profile", "full", "--dry-run"])
        )
        stage = next(s for s in run_pipeline.build_stages(cfg) if s.name == "figures")
        rels = {
            p.relative_to(cfg.paper_dir).as_posix()
            for p in stage.expected_outputs
        }

        expected = {"figure_manifest.json"}
        for stem in ("fig1", "fig2", "fig3", "fig4", "fig5", "fig6"):
            for kind in ("pdf", "svg", "png", "tiff"):
                expected.add(f"figures/{kind}/{stem}.{kind}")
        assert rels == expected

    def test_si_command_and_expected_outputs_cover_baseline_tables(self, run_pipeline):
        cfg = run_pipeline.resolve_config(
            run_pipeline.parse_args(["--profile", "full", "--dry-run"])
        )
        cmd = run_pipeline._cmd_si(cfg)
        assert "generate_si.py" in " ".join(cmd)
        assert "--output-dir" in cmd
        assert str(cfg.paper_dir) in cmd

        stage = next(s for s in run_pipeline.build_stages(cfg) if s.name == "si")
        names = {p.name for p in stage.expected_outputs}
        assert names == {
            "si_hyperparameters.tex",
            "si_hyperparameters.md",
            "si_sobol.tex",
            "si_sobol.md",
            "si_cases.tex",
            "si_cases.md",
            "si_multiple_testing.tex",
            "si_multiple_testing.md",
            "si_d2_policy.tex",
            "si_d2_policy.md",
        }
        assert stage.fresh_outputs is True

    def test_test_protocol_stage_materializes_mc_noise_inputs(self, run_pipeline):
        cfg = run_pipeline.resolve_config(
            run_pipeline.parse_args(["--profile", "full", "--dry-run"])
        )
        cmd = run_pipeline._cmd_test_protocol(cfg)
        assert "validate_test_protocol.py" in " ".join(cmd)
        assert str(cfg.sweep_dir) in cmd
        assert "--require-headline-source" in cmd

        stage = next(s for s in run_pipeline.build_stages(cfg) if s.name == "test_protocol")
        rels = {
            p.relative_to(cfg.sweep_dir / "analysis").as_posix()
            for p in stage.expected_outputs
        }
        assert rels == {
            "test_set_sufficiency.csv",
            "test_set_sufficiency.png",
            "cross_metric_ranks.csv",
            "cross_metric_rank_correlation.csv",
            "cross_metric_correlation.png",
            "method_robustness.csv",
            "method_robustness.png",
        }
        assert stage.fresh_outputs is True

    def test_case_validation_summary_stage_refreshes_generated_audit_doc(self, run_pipeline):
        cfg = run_pipeline.resolve_config(
            run_pipeline.parse_args(["--profile", "full", "--dry-run"])
        )
        note_cmd = run_pipeline._cmd_case_validation_note_sources(cfg)
        assert "update_case_validation_note_sources.py" in " ".join(note_cmd)
        assert "--write" in note_cmd
        assert "--output" in note_cmd
        assert "case_validation_note_sources.csv" in note_cmd[note_cmd.index("--output") + 1]

        cmd = run_pipeline._cmd_case_validation_summary(cfg)
        assert "generate_case_validation_summary.py" in " ".join(cmd)
        assert "--note-sources-csv" in cmd
        assert "case_validation_note_sources.csv" in cmd[cmd.index("--note-sources-csv") + 1]
        assert "--output" in cmd
        assert "--strict" in cmd
        assert "CASE_VALIDATION_SUMMARY.md" in cmd[cmd.index("--output") + 1]

        note_stage = next(
            s for s in run_pipeline.build_stages(cfg)
            if s.name == "case_validation_note_sources"
        )
        assert note_stage.expected_outputs == [
            run_pipeline.ROOT / "docs" / "case_validation_note_sources.csv"
        ]
        assert note_stage.fresh_outputs is True

        stage = next(
            s for s in run_pipeline.build_stages(cfg)
            if s.name == "case_validation_summary"
        )
        assert stage.expected_outputs == [run_pipeline.ROOT / "docs" / "CASE_VALIDATION_SUMMARY.md"]
        assert stage.fresh_outputs is True

    def test_sweep_audit_declares_full_grid_contract(self, run_pipeline):
        cfg = run_pipeline.resolve_config(
            run_pipeline.parse_args(["--profile", "full", "--dry-run"])
        )
        cmd = run_pipeline._cmd_sweep_audit(cfg)
        assert "audit_sweep_integrity.py" in " ".join(cmd)
        assert "--expected-cases" in cmd
        cases = cmd[cmd.index("--expected-cases") + 1].split(",")
        assert len(cases) == 34
        assert "--expected-methods" in cmd
        methods = cmd[cmd.index("--expected-methods") + 1].split(",")
        assert len(methods) in {18, 20}
        assert "--expected-method-families" in cmd
        families = cmd[cmd.index("--expected-method-families") + 1].split(",")
        assert len(families) == len(methods)
        assert all(":" in item for item in families)
        assert "--expected-budgets" in cmd
        bidx = cmd.index("--expected-budgets")
        assert cmd[bidx + 1: bidx + 5] == ["5", "10", "20", "50"]
        assert "--expected-repeats" in cmd
        assert cmd[cmd.index("--expected-repeats") + 1] == "30"
        assert "--expected-n-test" in cmd
        assert cmd[cmd.index("--expected-n-test") + 1] == "1000"
        assert "--expected-test-seed" in cmd
        assert cmd[cmd.index("--expected-test-seed") + 1] == "0"
        assert "--seed-strategy" in cmd
        assert cmd[cmd.index("--seed-strategy") + 1] == "case_repeat"
        assert "--seed-start" in cmd
        assert cmd[cmd.index("--seed-start") + 1] == "1000"
        assert "--min-ok-fraction" in cmd
        assert cmd[cmd.index("--min-ok-fraction") + 1] == "0.95"
        assert "--run-config-json" in cmd
        assert cmd[cmd.index("--run-config-json") + 1] == str(cfg.sweep_dir / "run_config.json")
        assert "--require-run-config" in cmd

    def test_manifest_includes_pool_and_budgets(self, run_pipeline):
        cfg = run_pipeline.resolve_config(
            run_pipeline.parse_args(["--profile", "full", "--dry-run"])
        )
        cmd = run_pipeline._cmd_manifest(cfg)
        assert "generate_run_manifest.py" in " ".join(cmd)
        assert "--pool" in cmd and "nature" in cmd
        # Manifest must echo the actual seed-start / strategy used so
        # reviewers can verify reproducibility.
        assert "--seed-strategy" in cmd
        assert "case_repeat" in cmd
        assert "--command" in cmd
        assert "--command-file" in cmd
        command_file = cmd[cmd.index("--command-file") + 1]
        assert command_file.endswith("pipeline_command_history.dry_run.txt")
        assert "--require-complete-archive" in cmd
        repro_command = cmd[cmd.index("--command") + 1]
        assert "run_pipeline.py" in repro_command
        assert "--profile" in repro_command
        assert "--confirm-full-run" in repro_command
        assert "--seed-start" in repro_command

    def test_manifest_uses_non_dry_run_command_history_for_real_freeze(self, run_pipeline):
        cfg = run_pipeline.resolve_config(
            run_pipeline.parse_args(["--profile", "full", "--confirm-full-run"])
        )

        cmd = run_pipeline._cmd_manifest(cfg)

        command_file = cmd[cmd.index("--command-file") + 1]
        assert command_file.endswith("pipeline_command_history.txt")
        assert not command_file.endswith("pipeline_command_history.dry_run.txt")

    def test_submission_ready_checks_manifest_numbers_and_figures(self, run_pipeline):
        cfg = run_pipeline.resolve_config(
            run_pipeline.parse_args(["--profile", "full", "--dry-run"])
        )
        cmd = run_pipeline._cmd_submission_ready(cfg)
        joined = " ".join(cmd)
        assert "check_submission_ready.py" in joined
        assert "--manifest" in cmd
        assert str(cfg.sweep_dir / "RUN_MANIFEST.json") in cmd
        assert "--numbers" in cmd
        assert str(cfg.paper_dir / "numbers.json") in cmd
        assert "--numbers-tex" in cmd
        assert str(cfg.paper_dir / "numbers.tex") in cmd
        numbers_cmd = run_pipeline._cmd_numbers(cfg)
        assert "--test-protocol-dir" in numbers_cmd
        assert str(cfg.sweep_dir / "analysis") in numbers_cmd
        assert "--require-results-csv" in numbers_cmd
        assert "--require-results-contract" in numbers_cmd
        assert "--require-submission-complete" in numbers_cmd
        assert "--require-headline-source" in numbers_cmd
        assert "--figure-manifest" in cmd
        assert str(cfg.paper_dir / "figure_manifest.json") in cmd
        assert "--figdata-dir" in cmd
        assert str(cfg.figdata_dir) in cmd
        assert "--case-validation-note-sources" in cmd
        assert (
            "case_validation_note_sources.csv"
            in cmd[cmd.index("--case-validation-note-sources") + 1]
        )
        assert "--case-validation-summary" in cmd
        assert "CASE_VALIDATION_SUMMARY.md" in cmd[cmd.index("--case-validation-summary") + 1]
        assert "--case-experiment-readiness" in cmd
        assert "case_experiment_readiness.csv" in cmd[cmd.index("--case-experiment-readiness") + 1]
        assert "--d1-dir" in cmd
        assert str(cfg.rr_dir) in cmd
        assert "--d2-dir" in cmd
        assert str(cfg.sc_dir) in cmd
        assert "--d3-dir" in cmd
        assert str(cfg.rr3_dir) in cmd
        assert "--test-protocol-dir" in cmd
        assert str(cfg.sweep_dir / "analysis") in cmd

    def test_figdata_command_uses_strict_d1_layout(self, run_pipeline):
        cfg = run_pipeline.resolve_config(
            run_pipeline.parse_args(["--profile", "full", "--dry-run"])
        )
        cmd = run_pipeline._cmd_figdata(cfg)

        assert "prepare_figdata.py" in " ".join(cmd)
        assert "--allow-legacy-d1-rank" not in cmd
        assert "--manifest" not in cmd
        assert "--expected-methods" in cmd
        assert "--require-headline-source" in cmd
        methods = cmd[cmd.index("--expected-methods") + 1].split(",")
        assert len(methods) in {18, 20}
        assert "botorch_sequential" not in methods

    def test_figdata_stage_cannot_read_stale_run_manifest(self, run_pipeline, tmp_path):
        cfg = run_pipeline.resolve_config(
            run_pipeline.parse_args([
                "--profile", "full",
                "--sweep-dir", str(tmp_path / "full_benchmark"),
                "--dry-run",
            ])
        )
        cfg.sweep_dir.mkdir(parents=True)
        (cfg.sweep_dir / "RUN_MANIFEST.json").write_text(
            '{"experiment":{"methods_declared":["botorch_sequential"]}}\n',
            encoding="utf-8",
        )

        cmd = run_pipeline._cmd_figdata(cfg)

        assert "--manifest" not in cmd
        methods = cmd[cmd.index("--expected-methods") + 1].split(",")
        assert "botorch_sequential" not in methods

    def test_sota_returns_none_on_windows(self, run_pipeline, monkeypatch):
        cfg = run_pipeline.resolve_config(run_pipeline.parse_args([
            "--profile", "full", "--with-sota", "--dry-run",
        ]))
        monkeypatch.setattr(run_pipeline.os, "name", "nt")
        assert run_pipeline._cmd_sota(cfg) is None

    def test_sota_skipped_without_optin(self, run_pipeline):
        cfg = run_pipeline.resolve_config(
            run_pipeline.parse_args(["--profile", "full", "--dry-run"])
        )
        assert run_pipeline._cmd_sota(cfg) is None


# ─── Dry-run integration ─────────────────────────────────────────────────
class TestDryRun:
    """End-to-end: ``--dry-run`` must succeed without spawning subprocesses."""

    def test_full_non_dry_run_requires_explicit_confirmation(self, run_pipeline, tmp_path, capsys):
        rc = run_pipeline.main([
            "--profile", "full",
            "--only", "validate",
            "--sweep-dir", str(tmp_path / "sweep"),
            "--logs-dir", str(tmp_path / "logs"),
        ])

        assert rc == 2
        err = capsys.readouterr().err
        assert "--confirm-full-run" in err

    def test_full_dry_run_does_not_require_confirmation(self, run_pipeline, tmp_path, capsys):
        rc = run_pipeline.main([
            "--profile", "full",
            "--dry-run",
            "--only", "validate",
            "--sweep-dir", str(tmp_path / "sweep"),
            "--logs-dir", str(tmp_path / "logs"),
        ])

        assert rc == 0
        out = capsys.readouterr().out
        assert "profile         : full" in out
        assert "[1] validate" in out

    def test_windows_direct_python_full_sweep_is_rejected(self, run_pipeline, monkeypatch):
        monkeypatch.setattr(run_pipeline.os, "name", "nt")
        cfg = run_pipeline.resolve_config(run_pipeline.parse_args([
            "--profile", "full",
            "--confirm-full-run",
        ]))
        stages = [
            stage
            for stage in run_pipeline.build_stages(cfg)
            if stage.name in {"validate", "sweep", "d1"}
        ]

        errors = run_pipeline.validate_windows_full_profile_boundary(cfg, stages)

        assert errors
        assert "Windows review workstations" in errors[0]
        assert "sweep" in errors[0]
        assert "d1" in errors[0]
        assert "bash scripts/run_pipeline.sh" in errors[0]

    def test_windows_direct_python_allows_stage1_and_dry_run(self, run_pipeline, monkeypatch):
        monkeypatch.setattr(run_pipeline.os, "name", "nt")
        full_cfg = run_pipeline.resolve_config(run_pipeline.parse_args([
            "--profile", "full",
            "--confirm-full-run",
        ]))
        stage1 = [
            stage
            for stage in run_pipeline.build_stages(full_cfg)
            if stage.number == 1
        ]
        dry_cfg = run_pipeline.resolve_config(run_pipeline.parse_args([
            "--profile", "full",
            "--confirm-full-run",
            "--dry-run",
        ]))
        full_stages = run_pipeline.build_stages(dry_cfg)

        assert run_pipeline.validate_windows_full_profile_boundary(full_cfg, stage1) == []
        assert run_pipeline.validate_windows_full_profile_boundary(dry_cfg, full_stages) == []

    def test_smoke_dry_run_returns_zero(self, run_pipeline, tmp_path, capsys):
        rc = run_pipeline.main([
            "--profile", "smoke",
            "--dry-run",
            "--sweep-dir", str(tmp_path / "sweep"),
            "--rr-dir", str(tmp_path / "rr"),
            "--sc-dir", str(tmp_path / "sc"),
            "--rr3-dir", str(tmp_path / "rr3"),
            "--figdata-dir", str(tmp_path / "figdata"),
            "--figures-dir", str(tmp_path / "figures"),
            "--paper-dir", str(tmp_path / "paper"),
            "--logs-dir", str(tmp_path / "logs"),
        ])
        assert rc == 0
        out = capsys.readouterr().out
        # Banner advertises the smoke profile.
        assert "profile         : smoke" in out
        assert "grid            :" in out
        assert "[0] smoke" in out
        assert "[1] validate" in out
        assert "[2] sweep" in out
        assert "[2] sweep_audit" in out
        assert "[3] report" in out
        assert "[4] d1" not in out
        # Each selected stage must be announced, but no subprocess
        # output should appear (because --dry-run short-circuits).
        assert "[dry-run]" not in out  # written to log file, not stdout

    def test_smoke_only_selects_stage_zero(self, run_pipeline, tmp_path, capsys):
        rc = run_pipeline.main([
            "--profile", "smoke",
            "--dry-run",
            "--only", "smoke",
            "--sweep-dir", str(tmp_path / "sweep"),
            "--logs-dir", str(tmp_path / "logs"),
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "[0] smoke" in out
        assert "[1] validate" not in out
        assert "[2] sweep" not in out

    def test_dry_run_writes_separate_command_history(self, run_pipeline, tmp_path):
        log_dir = tmp_path / "logs"
        rc = run_pipeline.main([
            "--profile", "smoke",
            "--dry-run",
            "--only", "validate",
            "--sweep-dir", str(tmp_path / "sweep"),
            "--logs-dir", str(log_dir),
        ])
        assert rc == 0
        logs = sorted(log_dir.glob("pipeline_*.log"))
        assert logs, "expected a pipeline_*.log file"
        text = logs[-1].read_text(encoding="utf-8")
        assert "Stage 1 — validate" in text
        assert "[dry-run]" in text
        assert "command_history:" in text

        assert not (log_dir / "pipeline_command_history.txt").exists()
        history = log_dir / "pipeline_command_history.dry_run.txt"
        assert history.is_file()
        history_text = history.read_text(encoding="utf-8")
        assert "pipeline_invocation\t" in history_text
        assert "--only validate" in history_text
        assert "validate\t" in history_text
        assert "validate_run_config.py" in history_text


# ─── run_stage postconditions ────────────────────────────────────────────
class TestRunStagePostconditions:
    def test_fresh_stage_rejects_stale_existing_output(self, run_pipeline, tmp_path):
        output = tmp_path / "artifact.txt"
        output.write_text("old\n", encoding="utf-8")
        stage = run_pipeline.Stage(
            number=99,
            name="noop",
            description="no-op stage",
            build_cmd=lambda cfg: [cfg.resolve_python(), "-c", "pass"],
            expected_outputs=[output],
            fresh_outputs=True,
        )
        cfg = self._cfg(run_pipeline, tmp_path)

        rc = run_pipeline.run_stage(
            stage,
            cfg,
            log_path=tmp_path / "logs" / "pipeline.log",
            continue_on_error=False,
        )

        assert rc == 1

    def test_postcondition_failure_stays_nonzero_with_continue_on_error(self, run_pipeline, tmp_path):
        output = tmp_path / "missing.txt"
        stage = run_pipeline.Stage(
            number=99,
            name="missing",
            description="missing output stage",
            build_cmd=lambda cfg: [cfg.resolve_python(), "-c", "pass"],
            expected_outputs=[output],
        )
        cfg = self._cfg(run_pipeline, tmp_path)

        rc = run_pipeline.run_stage(
            stage,
            cfg,
            log_path=tmp_path / "logs" / "pipeline.log",
            continue_on_error=True,
        )

        assert rc == 1

    def test_empty_expected_file_is_rejected(self, run_pipeline, tmp_path):
        output = tmp_path / "empty.csv"
        script = f"from pathlib import Path; Path({str(output)!r}).touch()"
        stage = run_pipeline.Stage(
            number=99,
            name="empty",
            description="empty output stage",
            build_cmd=lambda cfg: [cfg.resolve_python(), "-c", script],
            expected_outputs=[output],
        )
        cfg = self._cfg(run_pipeline, tmp_path)

        rc = run_pipeline.run_stage(
            stage,
            cfg,
            log_path=tmp_path / "logs" / "pipeline.log",
            continue_on_error=False,
        )

        assert rc == 1
        log_text = (tmp_path / "logs" / "pipeline.log").read_text(encoding="utf-8")
        assert "expected output files are empty" in log_text

    def test_empty_expected_directory_is_rejected(self, run_pipeline, tmp_path):
        output = tmp_path / "figs"
        script = f"from pathlib import Path; Path({str(output)!r}).mkdir(parents=True)"
        stage = run_pipeline.Stage(
            number=99,
            name="empty_dir",
            description="empty directory output stage",
            build_cmd=lambda cfg: [cfg.resolve_python(), "-c", script],
            expected_outputs=[output],
        )
        cfg = self._cfg(run_pipeline, tmp_path)

        rc = run_pipeline.run_stage(
            stage,
            cfg,
            log_path=tmp_path / "logs" / "pipeline.log",
            continue_on_error=False,
        )

        assert rc == 1
        log_text = (tmp_path / "logs" / "pipeline.log").read_text(encoding="utf-8")
        assert "expected output directories are empty" in log_text

    def test_expected_directory_with_content_is_accepted(self, run_pipeline, tmp_path):
        output = tmp_path / "figs"
        script = (
            f"from pathlib import Path; p=Path({str(output)!r}); "
            "p.mkdir(parents=True); (p / 'plot.png').write_text('png\\n', encoding='utf-8')"
        )
        stage = run_pipeline.Stage(
            number=99,
            name="dir_with_content",
            description="directory output stage",
            build_cmd=lambda cfg: [cfg.resolve_python(), "-c", script],
            expected_outputs=[output],
        )
        cfg = self._cfg(run_pipeline, tmp_path)

        rc = run_pipeline.run_stage(
            stage,
            cfg,
            log_path=tmp_path / "logs" / "pipeline.log",
            continue_on_error=False,
        )

        assert rc == 0

    def test_stale_output_failure_stays_nonzero_with_continue_on_error(self, run_pipeline, tmp_path):
        output = tmp_path / "artifact.txt"
        output.write_text("old\n", encoding="utf-8")
        stage = run_pipeline.Stage(
            number=99,
            name="stale",
            description="stale output stage",
            build_cmd=lambda cfg: [cfg.resolve_python(), "-c", "pass"],
            expected_outputs=[output],
            fresh_outputs=True,
        )
        cfg = self._cfg(run_pipeline, tmp_path)

        rc = run_pipeline.run_stage(
            stage,
            cfg,
            log_path=tmp_path / "logs" / "pipeline.log",
            continue_on_error=True,
        )

        assert rc == 1

    def test_fresh_stage_accepts_rewritten_output(self, run_pipeline, tmp_path):
        output = tmp_path / "artifact.txt"
        output.write_text("old\n", encoding="utf-8")
        time.sleep(0.01)
        script = f"from pathlib import Path; Path({str(output)!r}).write_text('new\\n', encoding='utf-8')"
        stage = run_pipeline.Stage(
            number=99,
            name="rewrite",
            description="rewrite stage",
            build_cmd=lambda cfg: [cfg.resolve_python(), "-c", script],
            expected_outputs=[output],
            fresh_outputs=True,
        )
        cfg = self._cfg(run_pipeline, tmp_path)

        rc = run_pipeline.run_stage(
            stage,
            cfg,
            log_path=tmp_path / "logs" / "pipeline.log",
            continue_on_error=False,
        )

        assert rc == 0
        assert output.read_text(encoding="utf-8") == "new\n"

    def test_resumable_stage_allows_existing_output_without_refresh(self, run_pipeline, tmp_path):
        output = tmp_path / "results.csv"
        output.write_text("old\n", encoding="utf-8")
        stage = run_pipeline.Stage(
            number=99,
            name="resume",
            description="resumable stage",
            build_cmd=lambda cfg: [cfg.resolve_python(), "-c", "pass"],
            expected_outputs=[output],
            fresh_outputs=False,
        )
        cfg = self._cfg(run_pipeline, tmp_path)

        rc = run_pipeline.run_stage(
            stage,
            cfg,
            log_path=tmp_path / "logs" / "pipeline.log",
            continue_on_error=False,
        )

        assert rc == 0

    def test_broken_console_flush_does_not_fail_stage_or_log_stream(
        self, run_pipeline, tmp_path, monkeypatch
    ):
        class FlushBrokenStdout:
            encoding = "utf-8"

            def __init__(self):
                self.writes = []

            def write(self, text):
                self.writes.append(text)
                return len(text)

            def flush(self):
                raise OSError(22, "Invalid argument")

        broken_stdout = FlushBrokenStdout()
        monkeypatch.setattr(run_pipeline.sys, "stdout", broken_stdout)
        monkeypatch.setattr(run_pipeline, "_CONSOLE_OUTPUT_BROKEN", False)

        stage = run_pipeline.Stage(
            number=99,
            name="stdout_broken",
            description="stage with child output and a broken parent stdout",
            build_cmd=lambda cfg: [
                cfg.resolve_python(),
                "-c",
                "print('child output survives in log')",
            ],
        )
        cfg = self._cfg(run_pipeline, tmp_path)
        log_path = tmp_path / "logs" / "pipeline.log"

        rc = run_pipeline.run_stage(
            stage,
            cfg,
            log_path=log_path,
            continue_on_error=False,
        )

        assert rc == 0
        log_text = log_path.read_text(encoding="utf-8")
        assert "child output survives in log" in log_text
        assert "[stage stdout_broken] exit=0" in log_text
        assert run_pipeline._CONSOLE_OUTPUT_BROKEN is True

    @staticmethod
    def _cfg(rp, tmp_path):
        return rp.resolve_config(rp.parse_args([
            "--profile", "smoke",
            "--sweep-dir", str(tmp_path / "sweep"),
            "--logs-dir", str(tmp_path / "logs"),
        ]))


# ─── BLAS thread pinning ─────────────────────────────────────────────────
class TestBlasPinning:
    def test_pin_blas_threads_sets_envvars(self, run_pipeline, monkeypatch):
        for var in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "BLIS_NUM_THREADS",
        ):
            monkeypatch.delenv(var, raising=False)
        run_pipeline.pin_blas_threads()
        assert run_pipeline.os.environ["OMP_NUM_THREADS"] == "1"
        assert run_pipeline.os.environ["OPENBLAS_NUM_THREADS"] == "1"
        assert run_pipeline.os.environ["MKL_NUM_THREADS"] == "1"

    def test_pin_blas_threads_overrides_existing_thread_caps(self, run_pipeline, monkeypatch):
        monkeypatch.setenv("OMP_NUM_THREADS", "8")
        run_pipeline.pin_blas_threads()
        assert run_pipeline.os.environ["OMP_NUM_THREADS"] == "1"
