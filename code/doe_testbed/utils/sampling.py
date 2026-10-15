"""
doe_testbed.utils.sampling
=============================
Unified interface for common sampling methods.

All methods return normalised sample matrices in [0, 1]^d, shape (n, d).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def sample(
    n: int,
    d: int,
    method: str = "lhs",
    seed: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    """General-purpose sampling interface.

    Parameters
    ----------
    n : int
        Number of samples.
    d : int
        Number of dimensions.
    method : str or callable
        Sampling method:
        'lhs'      — Optimised Latin hypercube (default)
        'sobol'    — Sobol low-discrepancy sequence
        'halton'   — Halton sequence
        'random'   — Uniform random
        'grid'     — Uniform grid (n rounded up to nearest perfect d-th power)
        'fullfact' — Full factorial design (n_levels^d)
        callable   — User-defined function accepting (n, d) returning (n, d) ndarray
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray, shape (n, d), values in [0, 1]^d
    """
    if callable(method):
        X = method(n, d, **kwargs)
        return np.asarray(X, dtype=float)

    method = method.lower()
    if method == "lhs":
        return _lhs(n, d, seed=seed, **kwargs)
    elif method == "sobol":
        return _sobol(n, d, seed=seed, **kwargs)
    elif method == "halton":
        return _halton(n, d, seed=seed)
    elif method == "random":
        rng = np.random.default_rng(seed)
        return rng.uniform(0, 1, size=(n, d))
    elif method == "grid":
        return _grid(n, d)
    elif method in ("fullfact", "full_factorial"):
        return _full_factorial(n, d)
    else:
        raise ValueError(
            f"Unknown sampling method '{method}'. "
            f"Available: lhs, sobol, halton, random, grid, fullfact, or callable."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Method implementations
# ─────────────────────────────────────────────────────────────────────────────


def _lhs(n: int, d: int, seed=None, criterion: str = "maximin") -> np.ndarray:
    """Optimised Latin hypercube sampling (scipy.stats.qmc).

    criterion : str
        'maximin'       — Maximise minimum inter-point distance (default, best space-filling)
        'centermaximin' — Centre each cell
        'correlation'   — Minimise off-diagonal correlation
        'random'        — No optimisation (fastest)
    """
    try:
        from scipy.stats.qmc import LatinHypercube

        sampler = LatinHypercube(
            d=d, seed=seed, optimization="random-cd" if criterion == "maximin" else None
        )
        return sampler.random(n)
    except Exception:
        # Fallback: manual LHS
        rng = np.random.default_rng(seed)
        X = np.empty((n, d))
        for j in range(d):
            perm = rng.permutation(n)
            X[:, j] = (perm + rng.uniform(size=n)) / n
        return X


def _sobol(n: int, d: int, seed=None, skip: int = 0) -> np.ndarray:
    """Sobol low-discrepancy sequence (scipy.stats.qmc).

    SciPy raises a ``UserWarning`` whenever ``n`` is not a power of two
    ("balance properties of Sobol points require n to be a power of 2").
    The scrambled Sobol' sequence is still a valid low-discrepancy
    sequence for any n, we simply lose the exact-balance guarantee at
    non-power-of-two sizes.  The benchmark uses ``n_test=1000`` by
    default, which fires this warning on every reference-set build and
    spams the sweep log with hundreds of copies; we silence just this
    one warning here, nothing else.
    """
    import warnings as _warnings
    try:
        from scipy.stats.qmc import Sobol

        sampler = Sobol(d=d, scramble=True, seed=seed)
        if skip > 0:
            sampler.fast_forward(skip)
        with _warnings.catch_warnings():
            _warnings.filterwarnings(
                "ignore",
                message="The balance properties of Sobol' points require",
                category=UserWarning,
            )
            return sampler.random(n)
    except Exception as e:
        raise ImportError(f"Sobol sampling requires scipy >= 1.7: {e}") from None


def _halton(n: int, d: int, seed=None) -> np.ndarray:
    """Halton sequence (scipy.stats.qmc)."""
    try:
        from scipy.stats.qmc import Halton

        sampler = Halton(d=d, scramble=True, seed=seed)
        return sampler.random(n)
    except Exception:
        raise ImportError("Halton sampling requires scipy >= 1.7") from None


def _grid(n: int, d: int) -> np.ndarray:
    """Uniform grid sampling.

    Each dimension gets the same number of equally spaced points;
    actual sample count is n_per_dim^d where n_per_dim = max(2, round(n^(1/d))).
    """
    n_per_dim = max(2, round(n ** (1.0 / d)))
    g = np.linspace(0, 1, n_per_dim)
    grids = np.meshgrid(*([g] * d), indexing="ij")
    X = np.column_stack([g.ravel() for g in grids])
    return X


def _full_factorial(n_levels: int, d: int) -> np.ndarray:
    """Full factorial design: n_levels levels, d factors, n_levels^d design points.

    Parameters
    ----------
    n_levels : int
        Number of levels per factor.
    d : int
        Number of factors (dimensions).
    """
    g = np.linspace(0, 1, n_levels)
    grids = np.meshgrid(*([g] * d), indexing="ij")
    return np.column_stack([g.ravel() for g in grids])


# ─────────────────────────────────────────────────────────────────────────────
# Convenience functions
# ─────────────────────────────────────────────────────────────────────────────


def lhs(n: int, d: int, seed: Optional[int] = None, **kwargs) -> np.ndarray:
    """Shortcut for Latin hypercube sampling."""
    return _lhs(n, d, seed=seed, **kwargs)


def sobol_sequence(n: int, d: int, seed: Optional[int] = None) -> np.ndarray:
    """Shortcut for Sobol sequence sampling."""
    return _sobol(n, d, seed=seed)
