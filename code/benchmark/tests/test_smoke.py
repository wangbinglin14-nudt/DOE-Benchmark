"""Smoke tests for the benchmark module: architecture sanity checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

import doe_testbed as db
from benchmark import (
    BenchmarkRunner,
    StaticDesignMethod,
    compute_all_metrics,
    default_method_pool,
    derive_seed,
)
from benchmark.samplers import (
    HaltonSampler,
    LHSSampler,
    OptimisedLHSSampler,
    RandomSampler,
    SobolSampler,
)
from benchmark.surrogates import (
    GPSurrogate,
    LinearSurrogate,
    PolynomialSurrogate,
    RandomForestSurrogate,
    RBFSurrogate,
)


# ─── Samplers ────────────────────────────────────────────────────────────
def test_all_samplers_produce_correct_shape():
    n, d = 50, 5
    for cls in [RandomSampler, LHSSampler, OptimisedLHSSampler, SobolSampler, HaltonSampler]:
        X = cls().sample(n, d, seed=42)
        assert X.shape == (n, d), f"{cls.__name__} returned {X.shape}"
        assert ((X >= 0) & (X <= 1)).all(), f"{cls.__name__} outside [0,1]"


def test_samplers_are_seeded():
    for cls in [LHSSampler, SobolSampler, HaltonSampler]:
        X1 = cls().sample(20, 3, seed=7)
        X2 = cls().sample(20, 3, seed=7)
        np.testing.assert_array_equal(X1, X2)


# ─── Surrogates ──────────────────────────────────────────────────────────
def test_surrogates_fit_predict_shape():
    rng = np.random.default_rng(0)
    X_train = rng.random((30, 4))
    Y_train = rng.random((30, 2))
    X_test = rng.random((10, 4))
    for cls in [GPSurrogate, RBFSurrogate, RandomForestSurrogate, LinearSurrogate, PolynomialSurrogate]:
        s = cls()
        s.fit(X_train, Y_train)
        mu, sd = s.predict(X_test)
        assert mu.shape == (10, 2), f"{cls.__name__} mean shape wrong: {mu.shape}"
        if sd is not None:
            assert sd.shape == (10, 2), f"{cls.__name__} std shape wrong: {sd.shape}"


# ─── Metrics ─────────────────────────────────────────────────────────────
def test_metrics_perfect_prediction():
    rng = np.random.default_rng(1)
    Y = rng.random((100, 3))
    out = compute_all_metrics(Y, Y)
    assert out["rmse"] < 1e-12
    assert out["mae"] < 1e-12
    assert abs(out["r2"] - 1.0) < 1e-9


def test_metrics_constant_baseline():
    rng = np.random.default_rng(2)
    Y_true = rng.random((100, 1))
    Y_pred = np.full_like(Y_true, Y_true.mean())
    out = compute_all_metrics(Y_true, Y_pred)
    # Constant prediction = mean → R² = 0
    assert abs(out["r2"]) < 0.05


# ─── End-to-end smoke ────────────────────────────────────────────────────
def test_static_design_method_end_to_end():
    case = db.get_case("cantilever_beam_5d")
    X_test = case.sample(200, method="sobol", seed=0)
    Y_test = case.evaluate(X_test)
    method = StaticDesignMethod(LHSSampler(), GPSurrogate(), name="lhs+gp")
    Y_pred, Y_std, info = method.run(case, n_train=30, X_test=X_test, seed=42)
    assert Y_pred.shape == Y_test.shape
    metrics = compute_all_metrics(Y_test, Y_pred, Y_std)
    assert metrics["rmse"] >= 0
    assert "t_fit_s" in info


def test_runner_micro_sweep():
    """Minimal runner sweep: 1 case × 2 methods × 1 budget × 2 repeats."""
    methods = [
        StaticDesignMethod(LHSSampler(), LinearSurrogate(), name="lhs+linear"),
        StaticDesignMethod(SobolSampler(), GPSurrogate(), name="sobol+gp"),
    ]
    runner = BenchmarkRunner(
        case_ids=["cantilever_beam_5d"],
        methods=methods,
        budgets=[5],
        n_repeats=2,
        n_test=200,
        verbose=False,
    )
    result = runner.run()
    assert isinstance(result.df, pd.DataFrame)
    assert len(result.df) == 1 * 2 * 1 * 2
    ok = (result.df["status"] == "ok").sum()
    assert ok == len(result.df), f"Expected all OK; got {ok}/{len(result.df)}"


def test_default_method_pool():
    pool = default_method_pool()
    assert len(pool) >= 8
    names = [m.name for m in pool]
    assert len(set(names)) == len(names), "duplicate method names"


def test_method_pool_sizes_documented():
    """Pool sizes must match the docstrings to avoid silent drift."""
    from benchmark.methods import default_method_pool, extended_method_pool, full_method_pool
    assert len(default_method_pool()) == 10
    assert len(extended_method_pool()) == 15
    assert len(full_method_pool()) == 18
    # No duplicate names across any pool
    for pool_fn in [default_method_pool, extended_method_pool, full_method_pool]:
        names = [m.name for m in pool_fn()]
        assert len(set(names)) == len(names), f"{pool_fn.__name__}: duplicate names"


# ─── Streaming-CSV schema evolution ─────────────────────────────────────
def test_stream_append_evolves_schema(tmp_path):
    """Different rows can have different keys; the streaming CSV must
    remain parseable by pandas at all times."""
    csv_path = tmp_path / "evolving.csv"

    # Write a row with the "narrow" schema
    BenchmarkRunner._stream_append_row(csv_path, {"a": 1, "b": 2})
    df = pd.read_csv(csv_path)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 1

    # Add a row with the same schema → no rewrite
    BenchmarkRunner._stream_append_row(csv_path, {"a": 3, "b": 4})
    df = pd.read_csv(csv_path)
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2

    # Add a row that introduces a new column "c"
    BenchmarkRunner._stream_append_row(csv_path, {"a": 5, "b": 6, "c": 7})
    df = pd.read_csv(csv_path)
    assert set(df.columns) == {"a", "b", "c"}
    assert len(df) == 3
    # Earlier rows should have NaN in column c
    assert pd.isna(df.loc[0, "c"])
    assert pd.isna(df.loc[1, "c"])
    assert df.loc[2, "c"] == 7

    # Add a row missing column "b" but introducing "d"
    BenchmarkRunner._stream_append_row(csv_path, {"a": 8, "c": 9, "d": 10})
    df = pd.read_csv(csv_path)
    assert set(df.columns) >= {"a", "b", "c", "d"}
    assert len(df) == 4
    assert df.loc[3, "a"] == 8
    assert pd.isna(df.loc[3, "b"])
    assert df.loc[3, "d"] == 10


def test_stream_append_consistent_with_runner(tmp_path):
    """Ensure that the streaming CSV from a real benchmark run is
    parseable mid-run on a 2-case sweep where the second case has
    extra reliability-metric columns."""
    methods = [StaticDesignMethod(LHSSampler(), LinearSurrogate(), name="lhs+linear")]
    runner = BenchmarkRunner(
        case_ids=["airfoil_stall_2d", "cantilever_beam_5d"],  # one with failure, one without
        methods=methods,
        budgets=[5],
        n_repeats=1,
        n_test=100,
        verbose=False,
        output_dir=tmp_path,
    )
    runner.run()
    df = pd.read_csv(tmp_path / "results.csv")
    assert len(df) == 2
    assert (df["status"] == "ok").all()


# ─── Seed derivation ─────────────────────────────────────────────────────
def test_derive_seed_case_repeat_independence():
    """Different (case, repeat) cells must yield distinct seeds, and the
    function must be deterministic / pure."""
    s1 = derive_seed("cantilever_beam_5d", 0)
    s2 = derive_seed("cantilever_beam_5d", 1)
    s3 = derive_seed("airfoil_stall_2d", 0)
    s4 = derive_seed("cantilever_beam_5d", 0)  # repeat call

    assert s1 != s2, "different repeats must produce different seeds"
    assert s1 != s3, "different cases must produce different seeds"
    assert s1 == s4, "derive_seed must be deterministic (pure)"
    # All seeds within signed-32-bit positive range (numpy / scipy compat)
    assert all(0 <= s < 2 ** 31 for s in (s1, s2, s3, s4))


def test_derive_seed_legacy_matches_old_behaviour():
    """The legacy strategy must reproduce ``seed_start + r``."""
    assert derive_seed("foo", 0, strategy="legacy", seed_start=1000) == 1000
    assert derive_seed("foo", 5, strategy="legacy", seed_start=1000) == 1005
    assert derive_seed("bar", 0, strategy="legacy", seed_start=1000) == 1000
    assert derive_seed("foo", 7, strategy="legacy", seed_start=42) == 49


def test_runner_uses_paired_seeds_across_methods(tmp_path):
    """All methods within the same (case, repeat) must share the seed
    so that paired-sample tests (Friedman / Wilcoxon) remain valid."""
    methods = [
        StaticDesignMethod(LHSSampler(), LinearSurrogate(), name="lhs+linear"),
        StaticDesignMethod(SobolSampler(), LinearSurrogate(), name="sobol+linear"),
    ]
    runner = BenchmarkRunner(
        case_ids=["cantilever_beam_5d"],
        methods=methods,
        budgets=[5],
        n_repeats=2,
        n_test=100,
        verbose=False,
        output_dir=tmp_path,
        seed_strategy="case_repeat",
    )
    runner.run()
    df = pd.read_csv(tmp_path / "results.csv")
    # Within (case, repeat), all methods share a seed
    by_repeat = df.groupby("repeat")["seed"].nunique()
    assert (by_repeat == 1).all(), "methods within same repeat must share seed"
    # Across repeats, seeds differ
    assert df["seed"].nunique() == 2


def test_runner_legacy_strategy_reproduces_old_seeds(tmp_path):
    """Running with strategy='legacy' must give seed = seed_start + r."""
    methods = [StaticDesignMethod(LHSSampler(), LinearSurrogate(), name="lhs+linear")]
    runner = BenchmarkRunner(
        case_ids=["cantilever_beam_5d"],
        methods=methods,
        budgets=[5],
        n_repeats=3,
        n_test=100,
        verbose=False,
        output_dir=tmp_path,
        seed_strategy="legacy",
        seed_start=1000,
    )
    runner.run()
    df = pd.read_csv(tmp_path / "results.csv")
    assert sorted(df["seed"].tolist()) == [1000, 1001, 1002]


def test_runner_rejects_unknown_seed_strategy(tmp_path):
    import pytest
    methods = [StaticDesignMethod(LHSSampler(), LinearSurrogate(), name="lhs+linear")]
    with pytest.raises(ValueError, match="seed_strategy"):
        BenchmarkRunner(
            case_ids=["cantilever_beam_5d"],
            methods=methods,
            budgets=[5],
            n_repeats=1,
            seed_strategy="banana",
        )
