"""Tests for doe_testbed.utils (sampling, cache) and registry query API."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import doe_testbed as db
from doe_testbed.utils.sampling import (
    _full_factorial,
    _grid,
    lhs,
    sample,
    sobol_sequence,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Sampling
# ═══════════════════════════════════════════════════════════════════════════════


class TestSampling:
    """Test all sampling methods return correct shapes and bounds."""

    @pytest.mark.parametrize("method", ["lhs", "sobol", "halton", "random"])
    def test_basic_methods(self, method):
        X = sample(20, 4, method=method, seed=42)
        assert X.shape == (20, 4)
        assert np.all(X >= 0) and np.all(X <= 1)

    def test_lhs_shortcut(self):
        X = lhs(15, 3, seed=0)
        assert X.shape == (15, 3)

    def test_sobol_shortcut(self):
        X = sobol_sequence(16, 3, seed=0)
        assert X.shape == (16, 3)

    def test_grid(self):
        X = _grid(100, 2)
        assert X.shape[1] == 2
        assert np.all(X >= 0) and np.all(X <= 1)

    def test_full_factorial(self):
        X = _full_factorial(3, 2)
        assert X.shape == (9, 2)  # 3^2

    def test_callable_method(self):
        def my_sampler(n, d):
            return np.full((n, d), 0.5)

        X = sample(10, 3, method=my_sampler)
        assert X.shape == (10, 3)
        assert np.allclose(X, 0.5)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown sampling method"):
            sample(10, 3, method="nonexistent")

    def test_seed_reproducibility(self):
        X1 = sample(20, 4, method="lhs", seed=123)
        X2 = sample(20, 4, method="lhs", seed=123)
        np.testing.assert_array_equal(X1, X2)

    def test_grid_method_via_sample(self):
        X = sample(100, 2, method="grid")
        assert X.shape[1] == 2

    def test_fullfact_method_via_sample(self):
        X = sample(3, 2, method="fullfact")
        assert X.shape == (9, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# Cache
# ═══════════════════════════════════════════════════════════════════════════════


class TestCacheManager:
    """Test CacheManager store / get / clear / list_cached."""

    def test_roundtrip(self, tmp_path):
        from doe_testbed.utils.cache import CacheManager

        cm = CacheManager(cache_dir=tmp_path)
        X = np.random.default_rng(0).uniform(size=(10, 3))
        Y = np.random.default_rng(1).uniform(size=(10, 2))

        # Not cached yet
        assert cm.get("test_case", 10, 0, "default") is None

        # Store
        cm.store("test_case", 10, 0, "default", X, Y)

        # Retrieve
        result = cm.get("test_case", 10, 0, "default")
        assert result is not None
        np.testing.assert_array_equal(result[0], X)
        np.testing.assert_array_equal(result[1], Y)

    def test_list_cached(self, tmp_path):
        from doe_testbed.utils.cache import CacheManager

        cm = CacheManager(cache_dir=tmp_path)
        X = np.zeros((5, 2))
        Y = np.zeros((5, 1))
        cm.store("case_a", 5, 0, "default", X, Y)
        cm.store("case_b", 5, 0, "default", X, Y)

        items = cm.list_cached()
        assert len(items) == 2

    def test_clear_all(self, tmp_path):
        from doe_testbed.utils.cache import CacheManager

        cm = CacheManager(cache_dir=tmp_path)
        X, Y = np.zeros((5, 2)), np.zeros((5, 1))
        cm.store("case_a", 5, 0, "default", X, Y)
        cm.store("case_b", 5, 0, "default", X, Y)

        count = cm.clear()
        assert count == 2
        assert cm.list_cached() == []

    def test_clear_by_case_id(self, tmp_path):
        from doe_testbed.utils.cache import CacheManager

        cm = CacheManager(cache_dir=tmp_path)
        X, Y = np.zeros((5, 2)), np.zeros((5, 1))
        cm.store("case_a", 5, 0, "default", X, Y)
        cm.store("case_b", 5, 0, "default", X, Y)

        count = cm.clear("case_a")
        assert count == 1
        assert len(cm.list_cached()) == 1

    def test_clear_empty_dir(self, tmp_path):
        from doe_testbed.utils.cache import CacheManager

        cm = CacheManager(cache_dir=tmp_path / "nonexistent")
        assert cm.clear() == 0

    def test_list_empty_dir(self, tmp_path):
        from doe_testbed.utils.cache import CacheManager

        cm = CacheManager(cache_dir=tmp_path / "nonexistent")
        assert cm.list_cached() == []

    def test_env_override(self, tmp_path, monkeypatch):
        from doe_testbed.utils.cache import _cache_dir

        monkeypatch.setenv("DOE_BENCHMARK_CACHE_DIR", str(tmp_path / "custom"))
        assert _cache_dir() == tmp_path / "custom"


# ═══════════════════════════════════════════════════════════════════════════════
# Registry query API
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegistry:
    """Test registry filtering and utility functions."""

    def test_list_cases_all(self):
        cases = db.list_cases()
        assert len(cases) == 34

    def test_list_cases_return_ids(self):
        ids = db.list_cases(return_ids=True)
        assert isinstance(ids, list)
        assert all(isinstance(i, str) for i in ids)
        assert len(ids) == 34

    def test_list_cases_domain_filter(self):
        aero = db.list_cases(domain="aerodynamics", return_ids=True)
        assert "wing_vlm_6d" in aero
        assert "cantilever_beam_5d" not in aero

    def test_list_cases_dim_range(self):
        small = db.list_cases(dim_input_range=(3, 4), return_ids=True)
        for cid in small:
            c = db.get_case(cid)
            assert 3 <= c.meta.dim_input <= 4

    def test_list_cases_failure_boundary(self):
        fb = db.list_cases(has_failure_boundary=True, return_ids=True)
        assert len(fb) >= 10

    def test_list_cases_multifidelity(self):
        mf = db.list_cases(supports_multifidelity=True, return_ids=True)
        assert len(mf) == 5

    def test_list_cases_cost_level(self):
        l0 = db.list_cases(cost_level="L0", return_ids=True)
        assert len(l0) >= 20

    def test_list_cases_vs_scenarios(self):
        vs_g = db.list_cases(applicable_scenarios=["VS-G"], return_ids=True)
        assert len(vs_g) >= 5

    def test_list_cases_response_topology(self):
        mono = db.list_cases(response_topology=["smooth_monotone"], return_ids=True)
        assert len(mono) >= 5

    def test_list_cases_max_eval_time(self):
        fast = db.list_cases(max_eval_time_ms=1.0, return_ids=True)
        assert len(fast) >= 10

    def test_list_cases_tags(self):
        tagged = db.list_cases(tags=["PDE"], return_ids=True)
        assert isinstance(tagged, list)

    def test_get_case_invalid_raises(self):
        with pytest.raises(KeyError, match="not found"):
            db.get_case("nonexistent_case_999")

    def test_get_minimum_covering_set(self):
        from doe_testbed.registry.registry import get_minimum_covering_set

        covering = get_minimum_covering_set()
        all_vs = set()
        for c in covering:
            all_vs |= set(c.meta.applicable_scenarios)
        assert all_vs >= {"VS-A", "VS-B", "VS-C", "VS-D", "VS-E", "VS-F", "VS-G", "VS-H"}

    def test_describe_case(self):
        from doe_testbed.registry.registry import describe_case

        desc = describe_case("cantilever_beam_5d")
        assert "cantilever_beam_5d" in desc

    def test_print_cases_table(self, capsys):
        from doe_testbed.registry.registry import print_cases_table

        print_cases_table()
        captured = capsys.readouterr()
        assert "cantilever_beam_5d" in captured.out
        assert "Total:" in captured.out

    def test_print_cases_table_empty(self, capsys):
        from doe_testbed.registry.registry import print_cases_table

        print_cases_table(metas=[])
        captured = capsys.readouterr()
        assert "no matching cases" in captured.out


# ═══════════════════════════════════════════════════════════════════════════════
# CLI (via function calls, not subprocess)
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════════


class TestVisualization:
    """Test visualization utilities with matplotlib Agg backend."""

    @pytest.fixture(autouse=True)
    def _use_agg(self):
        import matplotlib

        matplotlib.use("Agg")

    def test_plot_response_surface(self):
        from doe_testbed.utils.visualization import plot_response_surface

        case = db.get_case("cantilever_beam_5d")
        fig = plot_response_surface(case, dim_x=0, dim_y=1, n_grid=5, show=False)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_with_custom_fixed_values(self):
        from doe_testbed.utils.visualization import plot_response_surface

        case = db.get_case("cantilever_beam_5d")
        fig = plot_response_surface(
            case,
            dim_x=0,
            dim_y=3,
            fixed_values={2: 0.3, 4: 0.7},
            output_idx=1,
            title="Custom",
            n_grid=5,
            show=False,
        )
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_save(self, tmp_path):
        from doe_testbed.utils.visualization import plot_response_surface

        case = db.get_case("cantilever_beam_5d")
        out = str(tmp_path / "test_fig.png")
        fig = plot_response_surface(case, n_grid=5, show=False, save_path=out)
        assert Path(out).exists()
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_save_figure_creates_dirs(self, tmp_path):
        import matplotlib.pyplot as plt

        from doe_testbed.utils.visualization import save_figure

        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        out = str(tmp_path / "sub" / "dir" / "fig.png")
        save_figure(fig, out)
        assert Path(out).exists()
        plt.close(fig)


class TestCLI:
    """Test CLI commands by calling the handler functions directly."""

    def test_cmd_list(self, capsys):
        from doe_testbed.cli import cmd_list

        args = type("Args", (), {})()
        cmd_list(args)
        captured = capsys.readouterr()
        assert "DOE Testbed" in captured.out
        assert "cantilever_beam_5d" in captured.out

    def test_cmd_info(self, capsys):
        from doe_testbed.cli import cmd_info

        args = type("Args", (), {"case_id": "cantilever_beam_5d"})()
        cmd_info(args)
        captured = capsys.readouterr()
        assert "cantilever_beam_5d" in captured.out
        assert "Input variables" in captured.out

    def test_cmd_check_single(self, capsys):
        from doe_testbed.cli import cmd_check

        args = type("Args", (), {"case_id": "cantilever_beam_5d"})()
        cmd_check(args)
        captured = capsys.readouterr()
        assert "PASS" in captured.out

    def test_cmd_coverage(self, capsys):
        from doe_testbed.cli import cmd_coverage

        args = type("Args", (), {})()
        cmd_coverage(args)
        captured = capsys.readouterr()
        assert "VS-System Coverage Matrix" in captured.out
        assert "Dimension buckets" in captured.out
        assert "Cost buckets" in captured.out

    def test_main_no_args(self, capsys):
        import sys

        from doe_testbed.cli import main

        old_argv = sys.argv
        sys.argv = ["doe-testbed"]
        try:
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
        finally:
            sys.argv = old_argv
