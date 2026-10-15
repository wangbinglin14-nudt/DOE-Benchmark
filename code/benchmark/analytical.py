"""
benchmark.analytical
====================
Classical analytical test functions used as DOE benchmarks for 40+ years.

These functions are provided here purely for the **ranking-reversal**
experiment: comparing method rankings on analytical functions vs the
physics-grounded cases.  They are *not* part of the benchmark proper.

Each function is wrapped in a minimal :class:`AnalyticalCase` that
mimics the ``doe_testbed.BaseCase`` interface so the same
:class:`BenchmarkRunner` works on both.

References
----------
- Branin: Dixon, L.C.W., Szegö, G.P. (1978), *Towards Global Optimisation 2*.
- Hartmann 6-D: standard form (e.g. Picheny et al., 2013).
- Ishigami: Ishigami, T., Homma, T. (1990).
- Sobol' G-function: Saltelli, A. et al. (2008), *Global Sensitivity Analysis*.
- Friedman: Friedman, J.H. (1991).
- Welch et al. (1992) 20-D function.
- Wing-weight: Forrester, A. et al. (2008).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# Minimal stand-in dataclasses to mimic doe_testbed.CaseMeta interface
@dataclass
class _InputVar:
    name: str
    lower: float
    upper: float
    log_scale: bool = False


@dataclass
class _OutputVar:
    name: str
    unit: str = ""


@dataclass
class _Meta:
    case_id: str
    domain: str = "analytical"
    dim_input: int = 0
    dim_output: int = 1
    input_variables: List[_InputVar] = field(default_factory=list)
    output_variables: List[_OutputVar] = field(default_factory=list)
    fidelity_levels: tuple = ("default",)

    @property
    def input_names(self):
        return [v.name for v in self.input_variables]


# ─── AnalyticalCase wrapper ───────────────────────────────────────────────
class AnalyticalCase:
    """Mimics doe_testbed.BaseCase for analytical functions."""

    def __init__(self, meta: _Meta, fn) -> None:
        self.meta = meta
        self._fn = fn

    def evaluate(self, X, normalize_input: bool = True, **kwargs) -> np.ndarray:
        """Accepts X in [0,1]^d (when normalize_input=True) and returns Y."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if normalize_input:
            X_phys = self._denormalize(X)
        else:
            X_phys = X
        Y = self._fn(X_phys)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        return Y

    def _denormalize(self, X_norm: np.ndarray) -> np.ndarray:
        out = np.empty_like(X_norm, dtype=float)
        for j, var in enumerate(self.meta.input_variables):
            if var.log_scale:
                lo, hi = np.log10(var.lower), np.log10(var.upper)
                out[:, j] = 10 ** (lo + X_norm[:, j] * (hi - lo))
            else:
                out[:, j] = var.lower + X_norm[:, j] * (var.upper - var.lower)
        return out

    def sample(self, n: int, method: str = "sobol", seed: Optional[int] = None) -> np.ndarray:
        import warnings as _warnings
        from scipy.stats.qmc import Halton, LatinHypercube, Sobol
        d = self.meta.dim_input
        if method == "sobol":
            with _warnings.catch_warnings():
                _warnings.filterwarnings(
                    "ignore",
                    message="The balance properties of Sobol' points require",
                    category=UserWarning,
                )
                return Sobol(d=d, scramble=True, seed=seed).random(n)
        elif method == "lhs":
            return LatinHypercube(d=d, seed=seed).random(n)
        elif method == "halton":
            return Halton(d=d, scramble=True, seed=seed).random(n)
        else:
            rng = np.random.default_rng(seed)
            return rng.random((n, d))


# ─── Function definitions ────────────────────────────────────────────────
def _branin(X: np.ndarray) -> np.ndarray:
    """Branin 2-D function.  Domain: [-5,10]×[0,15]."""
    x1, x2 = X[:, 0], X[:, 1]
    a = 1.0
    b = 5.1 / (4 * np.pi ** 2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8 * np.pi)
    return a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s


def _hartmann6(X: np.ndarray) -> np.ndarray:
    """Hartmann 6-D function.  Domain: [0,1]^6."""
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14],
    ], dtype=float)
    P = 1e-4 * np.array([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381],
    ], dtype=float)
    Y = np.zeros(len(X))
    for i in range(4):
        inner = np.sum(A[i] * (X - P[i]) ** 2, axis=1)
        Y -= alpha[i] * np.exp(-inner)
    return Y


def _ishigami(X: np.ndarray) -> np.ndarray:
    """Ishigami 3-D function.  Domain: [-π, π]^3."""
    a, b = 7.0, 0.1
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    return np.sin(x1) + a * np.sin(x2) ** 2 + b * x3 ** 4 * np.sin(x1)


def _sobol_g(X: np.ndarray, a: Optional[np.ndarray] = None) -> np.ndarray:
    """Sobol' G-function.  Domain: [0,1]^d."""
    d = X.shape[1]
    if a is None:
        a = np.array([(j - 1) / 2.0 for j in range(1, d + 1)], dtype=float)
    return np.prod((np.abs(4 * X - 2) + a) / (1 + a), axis=1)


def _friedman(X: np.ndarray) -> np.ndarray:
    """Friedman 5-D function.  Domain: [0,1]^5."""
    return (
        10 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20 * (X[:, 2] - 0.5) ** 2
        + 10 * X[:, 3]
        + 5 * X[:, 4]
    )


def _wing_weight(X: np.ndarray) -> np.ndarray:
    """Wing weight 10-D function (Forrester, Sóbester, Keane 2008).

    Inputs (physical units, see denormalisation ranges below):
      Sw, Wfw, A, Lambda, q, lambda, tc, Nz, Wdg, Wp
    """
    Sw, Wfw, A, Lam, q, lam, tc, Nz, Wdg, Wp = (X[:, j] for j in range(10))
    Lam_rad = Lam * np.pi / 180.0
    return (
        0.036 * Sw ** 0.758 * Wfw ** 0.0035
        * (A / np.cos(Lam_rad) ** 2) ** 0.6
        * q ** 0.006 * lam ** 0.04
        * (100 * tc / np.cos(Lam_rad)) ** -0.3
        * (Nz * Wdg) ** 0.49
        + Sw * Wp
    )


# ─── Constructors ────────────────────────────────────────────────────────
def branin_2d() -> AnalyticalCase:
    meta = _Meta(
        case_id="branin_2d",
        dim_input=2, dim_output=1,
        input_variables=[
            _InputVar("x1", -5.0, 10.0),
            _InputVar("x2", 0.0, 15.0),
        ],
        output_variables=[_OutputVar("f")],
    )
    return AnalyticalCase(meta, _branin)


def hartmann_6d() -> AnalyticalCase:
    meta = _Meta(
        case_id="hartmann_6d",
        dim_input=6, dim_output=1,
        input_variables=[_InputVar(f"x{j+1}", 0.0, 1.0) for j in range(6)],
        output_variables=[_OutputVar("f")],
    )
    return AnalyticalCase(meta, _hartmann6)


def ishigami_3d() -> AnalyticalCase:
    meta = _Meta(
        case_id="ishigami_3d",
        dim_input=3, dim_output=1,
        input_variables=[_InputVar(f"x{j+1}", -np.pi, np.pi) for j in range(3)],
        output_variables=[_OutputVar("f")],
    )
    return AnalyticalCase(meta, _ishigami)


def sobol_g_8d() -> AnalyticalCase:
    meta = _Meta(
        case_id="sobol_g_8d",
        dim_input=8, dim_output=1,
        input_variables=[_InputVar(f"x{j+1}", 0.0, 1.0) for j in range(8)],
        output_variables=[_OutputVar("f")],
    )
    return AnalyticalCase(meta, _sobol_g)


def friedman_5d() -> AnalyticalCase:
    meta = _Meta(
        case_id="friedman_5d",
        dim_input=5, dim_output=1,
        input_variables=[_InputVar(f"x{j+1}", 0.0, 1.0) for j in range(5)],
        output_variables=[_OutputVar("f")],
    )
    return AnalyticalCase(meta, _friedman)


def wing_weight_10d() -> AnalyticalCase:
    """Wing-weight is itself an engineering proxy, but commonly used as analytical."""
    meta = _Meta(
        case_id="wing_weight_10d",
        dim_input=10, dim_output=1,
        input_variables=[
            _InputVar("Sw", 150.0, 200.0),
            _InputVar("Wfw", 220.0, 300.0),
            _InputVar("A", 6.0, 10.0),
            _InputVar("Lambda", -10.0, 10.0),
            _InputVar("q", 16.0, 45.0),
            _InputVar("lambda", 0.5, 1.0),
            _InputVar("tc", 0.08, 0.18),
            _InputVar("Nz", 2.5, 6.0),
            _InputVar("Wdg", 1700.0, 2500.0),
            _InputVar("Wp", 0.025, 0.08),
        ],
        output_variables=[_OutputVar("W", "lb")],
    )
    return AnalyticalCase(meta, _wing_weight)


# ─── Public registry ──────────────────────────────────────────────────────
ANALYTICAL_REGISTRY = {
    "branin_2d": branin_2d,
    "hartmann_6d": hartmann_6d,
    "ishigami_3d": ishigami_3d,
    "sobol_g_8d": sobol_g_8d,
    "friedman_5d": friedman_5d,
    "wing_weight_10d": wing_weight_10d,
}


def get_analytical(case_id: str) -> AnalyticalCase:
    if case_id not in ANALYTICAL_REGISTRY:
        raise KeyError(
            f"Unknown analytical case {case_id!r}; "
            f"available: {list(ANALYTICAL_REGISTRY)}"
        )
    return ANALYTICAL_REGISTRY[case_id]()


def list_analytical() -> List[str]:
    return list(ANALYTICAL_REGISTRY)
