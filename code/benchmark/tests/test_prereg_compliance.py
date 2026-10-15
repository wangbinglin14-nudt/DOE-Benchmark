"""Pre-registration compliance tests.

These tests freeze the v0.4.0-prereg protocol values into the test
suite so that any accidental drift in the codebase (a sneaky pool
addition, a budget tweak, a default seed-strategy change) is caught
in CI **before** it can corrupt a 30-repeat sweep.

If you intentionally amend the protocol, edit the constants below AND
add a row to ``paper/preregistration_amendments.md`` with a rationale.
The protocol version label is ``v0.4.0-prereg``; it is a version
identifier rather than a resolvable git tag, because the public git
history postdates the freeze (see ``paper/preregistration.md``).
"""
from __future__ import annotations

import re

import pytest

# ── Frozen pre-registration values (paper/preregistration.md §3) ────────
# Update only with an amendment.
PREREG_FREEZE_TAG = "v0.4.0-prereg"
PREREG_BUDGETS = [5, 10, 20, 50]
PREREG_N_REPEATS = 30
PREREG_N_TEST = 1000
PREREG_SEED_STRATEGY = "case_repeat"
PREREG_N_CASES = 34
PREREG_N_METHODS_FULL_POOL = 18      # static + sequential, no BoTorch
PREREG_N_METHODS_NATURE_POOL = 20    # full + 2 BoTorch
PREREG_BO_METHODS = {"botorch_qei", "botorch_saasbo"}
PREREG_SEQUENTIAL_METHODS = {"ego_ei", "max_variance", "akmcs"}
FORBIDDEN_LEGACY_METHODS = {"botorch_sequential"}


def _shell_cases_all(script_text: str) -> list[str]:
    match = re.search(r"CASES_ALL <<'EOF'.*?\n(?P<body>.*?)\nEOF", script_text, re.S)
    assert match, "launch_sweep_remote.sh missing CASES_ALL heredoc"
    return match.group("body").split()


def _powershell_cases(script_text: str) -> list[str]:
    match = re.search(r"\$Cases\s*=\s*@\((?P<body>.*?)\)", script_text, re.S)
    assert match, "launch_sweep_local_16gb.ps1 missing $Cases array"
    return re.findall(r'"([^"]+)"', match.group("body"))


class TestPreregCases:
    """The 34-case testbed cardinality is locked at the freeze."""

    def test_doe_testbed_has_exactly_34_cases(self):
        import doe_testbed as db
        cases = db.list_cases()
        assert len(cases) == PREREG_N_CASES, (
            f"doe_testbed exposes {len(cases)} cases; pre-reg locks "
            f"{PREREG_N_CASES}. Either add an amendment or revert the "
            f"case-registry change."
        )

    def test_dimension_distribution_unchanged(self):
        """The histogram of input dimensions is part of the protocol
        because Discovery 2's topology/dim stratification depends on it.
        """
        import doe_testbed as db
        dims = sorted(c.dim_input for c in db.list_cases())
        # Baseline taken at v0.4.0-prereg freeze
        expected = sorted(
            [3]              # 1×d=3
            + [4] * 6        # 6×d=4
            + [5] * 12       # 12×d=5
            + [6] * 7        # 7×d=6
            + [7]            # 1×d=7
            + [8] * 2        # 2×d=8
            + [9]            # 1×d=9
            + [10] * 2       # 2×d=10
            + [16]           # 1×d=16
            + [20]           # 1×d=20
        )
        assert dims == expected, (
            f"Dimension histogram changed:\n  got      {dims}\n"
            f"  expected {expected}"
        )


class TestPreregMethods:
    """The nature_pool() method count is locked at the freeze."""

    def test_full_pool_has_18_methods(self):
        from benchmark.methods import full_method_pool
        methods = full_method_pool()
        assert len(methods) == PREREG_N_METHODS_FULL_POOL, (
            f"full_method_pool() exposes {len(methods)} methods; "
            f"pre-reg locks {PREREG_N_METHODS_FULL_POOL}."
        )

    def test_nature_pool_has_20_methods_when_botorch_available(self):
        """If BoTorch is installed, nature_pool() must yield 20 methods."""
        try:
            import botorch  # noqa: F401
        except ImportError:
            pytest.skip(
                "BoTorch not installed - diagnostic fallback may expose 18 methods; "
                "formal Nature headline launch requires BoTorch/A0 and 20 methods"
            )
        from benchmark.methods import nature_pool
        methods = nature_pool()
        assert len(methods) == PREREG_N_METHODS_NATURE_POOL, (
            f"nature_pool() exposes {len(methods)} methods; "
            f"pre-reg locks {PREREG_N_METHODS_NATURE_POOL}."
        )

    def test_sequential_methods_present(self):
        """EGO, MaxVariance, AKMCS are required for the rigour analyses."""
        from benchmark.methods import full_method_pool
        names = {m.name for m in full_method_pool()}
        missing = PREREG_SEQUENTIAL_METHODS - names
        assert not missing, f"sequential methods missing: {missing}"

    def test_botorch_method_names_are_canonical(self):
        """The names 'botorch_qei' / 'botorch_saasbo' are referenced
        verbatim in paper/preregistration.md and downstream analysis
        scripts; renaming them silently breaks the pipeline.
        """
        try:
            from benchmark.methods import bo_method_pool
            methods = bo_method_pool()
        except ImportError:
            pytest.skip("BoTorch not installed")
        if not methods:
            pytest.skip("BoTorch pool empty (BoTorch not installed)")
        names = {m.name for m in methods}
        assert names <= PREREG_BO_METHODS, (
            f"BoTorch method names drifted: got {names}, "
            f"pre-reg locks subset of {PREREG_BO_METHODS}"
        )

    def test_legacy_method_names_absent_from_production_pools(self):
        from benchmark.methods import list_methods

        pools = ["default", "extended", "full", "bo", "nature", "sota", "mf"]
        observed = {pool: set(list_methods(pool)) for pool in pools}
        offenders = {
            pool: sorted(names & FORBIDDEN_LEGACY_METHODS)
            for pool, names in observed.items()
            if names & FORBIDDEN_LEGACY_METHODS
        }
        assert not offenders, f"legacy method names leaked into production pools: {offenders}"

    def test_preregistration_text_keeps_20_method_headline_no_fallback(self):
        from pathlib import Path

        text = (Path(__file__).resolve().parents[2] / "paper" / "preregistration.md").read_text(
            encoding="utf-8"
        )

        assert "$N = 20$ method ranks" in text
        assert "complete 20-method environment" in text
        assert "launch blocker" in text
        assert "degrades to 18 methods" not in text

    def test_preregistration_text_uses_current_server_pipeline(self):
        from pathlib import Path

        text = (Path(__file__).resolve().parents[2] / "paper" / "preregistration.md").read_text(
            encoding="utf-8"
        )

        assert "python scripts/check_server_launch_ready.py" in text
        assert "bash scripts/run_pipeline.sh" in text
        assert "--confirm-large-run --require-botorch" in text
        assert "snakemake --cores" not in text
        assert "docker run --rm" not in text


class TestPreregSeedStrategy:
    def test_case_repeat_strategy_exists(self):
        """The seed strategy 'case_repeat' must remain valid."""
        from benchmark.core import derive_seed
        # Should not raise
        s1 = derive_seed("cantilever_beam_5d", 0, strategy="case_repeat", seed_start=1000)
        s2 = derive_seed("cantilever_beam_5d", 0, strategy="case_repeat", seed_start=1000)
        assert s1 == s2, "seed derivation must be deterministic"

    def test_case_repeat_is_independent_per_case(self):
        from benchmark.core import derive_seed
        s_a = derive_seed("cantilever_beam_5d", 0, strategy="case_repeat", seed_start=1000)
        s_b = derive_seed("airfoil_stall_2d", 0, strategy="case_repeat", seed_start=1000)
        assert s_a != s_b, "different cases must yield different seeds"


class TestPreregValidator:
    """Smoke-test the validator script itself so it stays runnable."""

    def test_validator_accepts_compliant_config(self, tmp_path):
        from scripts.validate_run_config import main
        rc = main([
            "--pool", "nature",
            "--budgets", "5", "10", "20", "50",
            "--n-repeats", "30",
            "--n-test", "1000",
            "--seed-strategy", "case_repeat",
            "--cases", "ALL",
        ])
        assert rc == 0

    def test_validator_accepts_explicit_current_case_list(self):
        import doe_testbed as db
        from scripts.validate_run_config import main

        cases = sorted(case.case_id for case in db.list_cases())
        rc = main([
            "--pool", "nature",
            "--budgets", "5", "10", "20", "50",
            "--n-repeats", "30",
            "--n-test", "1000",
            "--seed-strategy", "case_repeat",
            "--cases", *cases,
        ])

        assert rc == 0

    def test_headline_launchers_validate_the_explicit_case_list(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        remote = (root / "scripts" / "launch_sweep_remote.sh").read_text(
            encoding="utf-8",
        )
        local = (root / "scripts" / "launch_sweep_local_16gb.ps1").read_text(
            encoding="utf-8",
        )

        assert "--cases ${CASES_FLAT}" in remote
        assert "--cases ALL" not in remote
        assert '"--cases"' in local
        assert '$ValidatorArgs += $Cases' in local
        assert '"--cases", "ALL"' not in local

    def test_headline_launcher_case_lists_match_current_registry(self):
        import doe_testbed as db
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        expected = sorted(case.case_id for case in db.list_cases())
        remote = (root / "scripts" / "launch_sweep_remote.sh").read_text(
            encoding="utf-8",
        )
        local = (root / "scripts" / "launch_sweep_local_16gb.ps1").read_text(
            encoding="utf-8",
        )

        assert sorted(_shell_cases_all(remote)) == expected
        assert sorted(_powershell_cases(local)) == expected

    def test_sota_launcher_protocol_matches_headline_case_and_seed_grid(self):
        import doe_testbed as db
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        expected = sorted(case.case_id for case in db.list_cases())
        sota = (root / "scripts" / "launch_sota_robustness.sh").read_text(
            encoding="utf-8",
        )

        assert sorted(_shell_cases_all(sota)) == expected
        assert "--cases ${CASES_FLAT}" in sota
        assert "SOTA launcher CASES_ALL does not match doe_testbed registry" in sota
        assert "--n-test 1000" in sota
        assert "--seed-strategy case_repeat" in sota
        assert "--seed-start 1000" in sota
        assert "benchmark/results/sota_robustness" in sota
        assert "SOTA robustness OUT_DIR must not be benchmark/results/full_benchmark" in sota
        assert "HEADLINE_DIR must not point to the SOTA robustness output directory" in sota
        assert "HEADLINE_DIR must not point to benchmark/results/mf_supplementary" in sota

    def test_mfbo_launcher_protocol_matches_multifidelity_registry(self):
        import doe_testbed as db
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        mf_cases = sorted(
            case.case_id
            for case in db.list_cases()
            if getattr(case, "supports_multifidelity", False)
        )
        mfbo = (root / "scripts" / "launch_mfbo.sh").read_text(
            encoding="utf-8",
        )

        assert len(mf_cases) == 5
        assert "--pool mf" in mfbo
        assert "supports_multifidelity" in mfbo
        assert '[[ "${MF_N_CASES}" -ne 5 ]]' in mfbo
        assert "--cases ${MF_CASES_FLAT}" in mfbo
        assert "--cases-filter mf" not in mfbo
        assert "--n-test 1000" in mfbo
        assert "--seed-strategy case_repeat" in mfbo
        assert "--seed-start 1000" in mfbo
        assert "benchmark/results/mf_supplementary" in mfbo
        assert "MF supplementary OUT_DIR must not be benchmark/results/full_benchmark" in mfbo
        assert "MF supplementary OUT_DIR must not point to benchmark/results/sota_robustness" in mfbo

    def test_validator_rejects_wrong_pool(self):
        from scripts.validate_run_config import main
        rc = main([
            "--pool", "extended",
            "--budgets", "5", "10", "20", "50",
            "--n-repeats", "30",
            "--n-test", "1000",
            "--seed-strategy", "case_repeat",
            "--cases", "ALL",
        ])
        assert rc == 1

    def test_validator_rejects_wrong_budgets(self):
        from scripts.validate_run_config import main
        rc = main([
            "--pool", "nature",
            "--budgets", "5", "10", "20",   # missing 50
            "--n-repeats", "30",
            "--n-test", "1000",
            "--seed-strategy", "case_repeat",
            "--cases", "ALL",
        ])
        assert rc == 1

    def test_validator_rejects_low_n_repeats(self):
        from scripts.validate_run_config import main
        rc = main([
            "--pool", "nature",
            "--budgets", "5", "10", "20", "50",
            "--n-repeats", "5",   # pilot, not headline
            "--n-test", "1000",
            "--seed-strategy", "case_repeat",
            "--cases", "ALL",
        ])
        assert rc == 1

    def test_validator_allow_deviation_writes_log(self, tmp_path):
        from scripts.validate_run_config import main
        log_dir = tmp_path / "out"
        rc = main([
            "--pool", "extended",
            "--budgets", "5", "10", "20", "50",
            "--n-repeats", "30",
            "--n-test", "1000",
            "--seed-strategy", "case_repeat",
            "--cases", "ALL",
            "--allow-deviation",
            "--output-dir", str(log_dir),
        ])
        assert rc == 0  # allowed
        log_path = log_dir / "preregistration_deviations.txt"
        assert log_path.exists()
        content = log_path.read_text(encoding="utf-8")
        assert "extended" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
