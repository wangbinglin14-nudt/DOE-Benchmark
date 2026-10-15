"""
doe_testbed.base.case
=======================
BaseCase abstract base class -- the unified interface contract for all cases.

Subclasses must implement:
    meta           : CaseMeta   (class attribute)
    _evaluate_core(X_phys, fidelity, **kwargs) -> np.ndarray

Subclasses may optionally override:
    _evaluate_gradient_core(X_phys, fidelity) -> np.ndarray
    plot_response_surface(...)   (for custom visualisation)
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from doe_testbed.base.meta import CaseMeta

# ─────────────────────────────────────────────────────────────────────────────
# Helpers: vectorised normalisation / denormalisation
# ─────────────────────────────────────────────────────────────────────────────


def _normalize_array(X_phys: np.ndarray, meta: CaseMeta) -> np.ndarray:
    """Physical matrix -> [0,1]^d normalised matrix (copy-safe).

    Vectorised per-column transform: linear for ordinary variables,
    base-10 logarithmic for ``log_scale=True`` ones (order of magnitude
    faster than ``np.vectorize`` over the scalar ``InputVariable``
    helpers, especially for the 10 000-point reference set).
    """
    X = np.empty_like(X_phys, dtype=float)
    for j, var in enumerate(meta.input_variables):
        col = X_phys[:, j]
        if var.log_scale:
            lo = np.log10(var.lower)
            hi = np.log10(var.upper)
            X[:, j] = (np.log10(col) - lo) / (hi - lo)
        else:
            X[:, j] = (col - var.lower) / (var.upper - var.lower)
    return X


def _denormalize_array(X_norm: np.ndarray, meta: CaseMeta) -> np.ndarray:
    """[0,1]^d normalised matrix -> physical matrix.

    Mirror of :func:`_normalize_array`; vectorised per column.
    """
    X = np.empty_like(X_norm, dtype=float)
    for j, var in enumerate(meta.input_variables):
        col = X_norm[:, j]
        if var.log_scale:
            lo = np.log10(var.lower)
            hi = np.log10(var.upper)
            X[:, j] = np.power(10.0, lo + col * (hi - lo))
        else:
            X[:, j] = var.lower + col * (var.upper - var.lower)
    return X


# ─────────────────────────────────────────────────────────────────────────────
# Bounds check
# ─────────────────────────────────────────────────────────────────────────────


def _check_bounds(X_norm: np.ndarray, case_id: str, check: bool) -> None:
    """Check that normalised inputs lie within [0,1]^d."""
    if not check:
        return
    out_of_bounds = np.any((X_norm < -1e-8) | (X_norm > 1.0 + 1e-8), axis=1)
    n_oob = int(out_of_bounds.sum())
    if n_oob == 0:
        return
    frac = n_oob / len(X_norm)
    msg = (
        f"[{case_id}] {n_oob}/{len(X_norm)} samples ({frac*100:.1f}%) "
        f"are outside the normalised bounds [0,1]^d."
    )
    if frac > 0.05:
        raise ValueError(msg + " Out-of-bounds fraction >5%; please check input.")
    warnings.warn(msg, UserWarning, stacklevel=4)


# ─────────────────────────────────────────────────────────────────────────────
# BaseCase
# ─────────────────────────────────────────────────────────────────────────────


class BaseCase(ABC):
    """Abstract base class for all DOE Testbed simulation cases.

    All cases expose a unified interface, hiding solver-specific details.

    Examples
    --------
    >>> import doe_testbed as db
    >>> case = db.get_case("airfoil_stall_2d")
    >>> X = case.sample(50, method='lhs', seed=42)  # normalised space (50, 4)
    >>> Y = case.evaluate(X)                         # physical-unit output (50, 2)
    >>> case.describe()
    """

    # Subclasses must define (class attribute)
    meta: CaseMeta

    # ── Must implement (subclass override) ────────────────────────────────

    @abstractmethod
    def _evaluate_core(
        self,
        X_phys: np.ndarray,
        fidelity: str = "default",
        **kwargs,
    ) -> np.ndarray:
        """Core physics simulation -- subclasses must implement.

        Parameters
        ----------
        X_phys : np.ndarray, shape (n, d)
            Physical-unit input matrix (already denormalised; units per meta.input_variables).
        fidelity : str
            Fidelity level, one of meta.fidelity_levels.

        Returns
        -------
        Y : np.ndarray, shape (n, m)
            Output response matrix (physical units).
            Rows for failed evaluations should be filled with np.nan (do not raise).

        Notes
        -----
        Implementation constraints:
          1. Must be a pure function (no randomness, no side effects).
          2. Must support batch input (n >= 1).
          3. Must not modify X_phys in-place.
        """

    # ── Public interface (user-facing) ──────────────────────────────────────

    def evaluate(
        self,
        X: Union[np.ndarray, list],
        fidelity: str = "default",
        normalize_input: bool = True,
        check_bounds: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """Main evaluation interface.

        Parameters
        ----------
        X : array-like, shape (n, d) or (d,)
            Input matrix.
            normalize_input=True (default): X in [0,1]^d normalised space;
            normalize_input=False: X in physical-unit space.
        fidelity : str
            Fidelity level, default 'default'.
        normalize_input : bool
            Whether to treat input as normalised, default True.
        check_bounds : bool
            Whether to check that inputs are within bounds, default True.

        Returns
        -------
        Y : np.ndarray, shape (n, m)
            Output response matrix (physical units).
            Single-sample input still returns (1, m), not (m,).

        Examples
        --------
        >>> X = case.sample(100, seed=0)
        >>> Y = case.evaluate(X)          # (100, m)
        >>> Y_single = case.evaluate(X[0])  # (1, m)
        """
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        if X_arr.ndim != 2 or X_arr.shape[1] != self.meta.dim_input:
            raise ValueError(
                f"[{self.meta.case_id}] Expected input shape (n, {self.meta.dim_input}), "
                f"got {X_arr.shape}."
                f"\nInput variables: {self.meta.input_names}"
            )
        if fidelity not in self.meta.fidelity_levels:
            raise ValueError(
                f"[{self.meta.case_id}] Unsupported fidelity='{fidelity}'; "
                f"available: {self.meta.fidelity_levels}"
            )
        if normalize_input:
            _check_bounds(X_arr, self.meta.case_id, check_bounds)
            X_phys = _denormalize_array(X_arr, self.meta)
        else:
            X_phys = X_arr.copy()

        Y = self._evaluate_core(X_phys, fidelity=fidelity, **kwargs)
        Y = np.asarray(Y, dtype=float)
        if Y.shape != (len(X_phys), self.meta.dim_output):
            raise RuntimeError(
                f"[{self.meta.case_id}] _evaluate_core returned shape {Y.shape}, "
                f"expected ({len(X_phys)}, {self.meta.dim_output})."
            )
        return Y

    def sample(
        self,
        n: int,
        method: str = "lhs",
        seed: Optional[int] = None,
        **kwargs,
    ) -> np.ndarray:
        """Generate design points in the normalised [0,1]^d space.

        Parameters
        ----------
        n : int
            Number of samples.
        method : str
            Sampling method: 'lhs' | 'sobol' | 'halton' | 'random' |
                      'grid' | 'fullfact' | callable
            Default 'lhs' (optimised Latin hypercube).
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        X : np.ndarray, shape (n, d), values in [0, 1]^d
        """
        from doe_testbed.utils.sampling import sample as _sample

        return _sample(n, self.meta.dim_input, method=method, seed=seed, **kwargs)

    def evaluate_and_sample(
        self,
        n: int,
        method: str = "lhs",
        seed: Optional[int] = None,
        fidelity: str = "default",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample and evaluate in one step, returning (X, Y).

        Equivalent to:
            X = case.sample(n, method, seed)
            Y = case.evaluate(X, fidelity)
        """
        X = self.sample(n, method=method, seed=seed)
        Y = self.evaluate(X, fidelity=fidelity)
        return X, Y

    def get_reference_set(
        self,
        n: int = 10000,
        seed: int = 0,
        fidelity: str = "default",
        use_cache: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Obtain a fixed standard reference test set (X_ref, Y_ref).

        The reference set is generated with a fixed Sobol sequence
        (uniform, low-discrepancy) for computing RMSE and other accuracy
        metrics on a common test set.

        Parameters
        ----------
        n : int
            Reference set size, default 10000.
        seed : int
            Fixed seed, default 0 (ensures identical reference set for all users).
        use_cache : bool
            Whether to use local cache (important for expensive cases).

        Returns
        -------
        X_ref : np.ndarray, shape (n, d), normalised space
        Y_ref : np.ndarray, shape (n, m), physical units
        """
        from doe_testbed.utils.cache import CacheManager

        mgr = CacheManager()

        if use_cache:
            cached = mgr.get(self.meta.case_id, n, seed, fidelity)
            if cached is not None:
                return cached

        X_ref = self.sample(n, method="sobol", seed=seed)
        Y_ref = self.evaluate(X_ref, fidelity=fidelity)

        if use_cache:
            mgr.store(self.meta.case_id, n, seed, fidelity, X_ref, Y_ref)
        return X_ref, Y_ref

    def normalize(self, X_phys: np.ndarray) -> np.ndarray:
        """Physical-unit input -> [0, 1]^d normalised space.

        Parameters
        ----------
        X_phys : np.ndarray, shape (n, d) or (d,)

        Returns
        -------
        X_norm : np.ndarray, shape (n, d) or (1, d)
        """
        X = np.asarray(X_phys, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return _normalize_array(X, self.meta)

    def denormalize(self, X_norm: np.ndarray) -> np.ndarray:
        """[0, 1]^d normalised space -> physical-unit input.

        Parameters
        ----------
        X_norm : np.ndarray, shape (n, d) or (d,)

        Returns
        -------
        X_phys : np.ndarray, shape (n, d) or (1, d)
        """
        X = np.asarray(X_norm, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return _denormalize_array(X, self.meta)

    def describe(self, verbose: bool = True) -> str:
        """Return a human-readable full description of the case.

        Parameters
        ----------
        verbose : bool
            If True, include Sobol indices and detailed scenario notes.

        Returns
        -------
        str
        """
        m = self.meta
        lines = [
            "=" * 72,
            f"  {m.case_id}  ·  {m.name}",
            "=" * 72,
            f"  Domain     : {m.domain}",
            f"  Governing  : {m.governing_equation}",
            f"  Solver     : {m.solver_library}  ({m.solver_details})",
            f"  Version    : {m.version}",
            "",
            "── Input variables " + "─" * 52,
        ]
        header = f"  {'name':12s} {'symbol':10s} {'range':22s} {'unit':8s} {'role':6s}"
        lines.append(header)
        lines.append("  " + "-" * 60)
        for v in m.input_variables:
            rng = f"[{v.lower}, {v.upper}]"
            log_mark = "(log)" if v.log_scale else ""
            lines.append(
                f"  {v.name:12s} {v.symbol:10s} {rng:22s} {v.unit:8s} " f"{v.role:6s} {log_mark}"
            )
        lines += [
            "",
            "── Output variables " + "─" * 51,
        ]
        for ov in m.output_variables:
            ft = ""
            if ov.failure_threshold is not None:
                ft = f"  failure: {ov.failure_direction} {ov.failure_threshold} {ov.unit}"
            lines.append(f"  {ov.name:12s} {ov.symbol:10s} {ov.description}{ft}")
        lines += [
            "",
            "── Response surface " + "─" * 51,
            f"  Topology   : {', '.join(m.response_topology)}",
            f"  Eff. dim   : {m.effective_dim or m.dim_input} / {m.dim_input}",
            f"  Important  : {', '.join(m.important_variables) or '(not computed)'}",
            f"  Failure bdy: {'Yes — ' + (m.failure_condition or '') if m.has_failure_boundary else 'No'}",
            "",
            "── VS scenarios " + "─" * 55,
        ]
        for scn in m.applicable_scenarios:
            note = m.scenario_notes.get(scn, "")
            lines.append(f"  {scn:6s} : {note}")
        lines += [
            "",
            "── Computational cost " + "─" * 49,
            f"  Cost level : {m.cost_level}",
        ]
        for fid, t in m.typical_eval_time_ms.items():
            lines.append(f"  {fid:10s} : {t:.1f} ms/eval")
        if verbose and m.sobol_S1:
            lines += ["", "── Sobol indices (S1 / ST) " + "─" * 44]
            for name in m.input_names:
                s1 = m.sobol_S1.get(name, 0.0)
                st = m.sobol_ST.get(name, 0.0) if m.sobol_ST else 0.0
                bar = "█" * int(st * 20)
                lines.append(f"  {name:12s}  S1={s1:.3f}  ST={st:.3f}  {bar}")
        if m.physics_references:
            lines += ["", "── References " + "─" * 57]
            for ref in m.physics_references:
                lines.append(f"  - {ref}")
        lines.append("=" * 72)
        text = "\n".join(lines)
        print(text)
        return text

    def plot_response_surface(
        self,
        x_dim: int = 0,
        y_dim: int = 1,
        fixed_values: Optional[Dict[str, float]] = None,
        n_grid: int = 50,
        output_idx: int = 0,
        fidelity: str = "default",
        ax=None,
        colormap: str = "viridis",
        title: Optional[str] = None,
    ):
        """Plot a 2-D response-surface slice (other dims fixed at 0.5).

        Parameters
        ----------
        x_dim, y_dim : int
            Dimension indices for the two plot axes.
        fixed_values : dict, optional
            Fixed values for other dimensions (normalised space); default 0.5.
        n_grid : int
            Grid points per dimension (n_grid^2 evaluations total).
        output_idx : int
            Which output to plot.
        ax : matplotlib.axes.Axes, optional
            If not provided, a new figure is created.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Plotting requires matplotlib: pip install matplotlib") from None

        # Build grid
        g = np.linspace(0, 1, n_grid)
        xx, yy = np.meshgrid(g, g)
        X_grid = np.full((n_grid * n_grid, self.meta.dim_input), 0.5)
        # Apply fixed_values
        if fixed_values:
            for k, val in fixed_values.items():
                if k in self.meta.input_names:
                    idx = self.meta.input_names.index(k)
                    X_grid[:, idx] = val
        X_grid[:, x_dim] = xx.ravel()
        X_grid[:, y_dim] = yy.ravel()

        Y_grid = self.evaluate(X_grid, fidelity=fidelity)
        Z = Y_grid[:, output_idx].reshape(n_grid, n_grid)

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))

        im = ax.contourf(xx, yy, Z, levels=30, cmap=colormap)
        plt.colorbar(im, ax=ax, label=self.meta.output_variables[output_idx].name)
        x_var = self.meta.input_variables[x_dim]
        y_var = self.meta.input_variables[y_dim]
        ax.set_xlabel(f"{x_var.name} (normalised)")
        ax.set_ylabel(f"{y_var.name} (normalised)")
        ax.set_title(
            title or f"{self.meta.case_id} — " f"{self.meta.output_variables[output_idx].name}"
        )
        return ax

    def check_implementation(self, verbose: bool = True) -> Dict[str, Any]:
        """Automated implementation conformance test (for CI/CD).

        Returns
        -------
        dict
            Pass/fail status for each check (True = pass, False = fail).
        """
        import time

        results: Dict[str, Any] = {}
        d, m = self.meta.dim_input, self.meta.dim_output

        # 1. Single sample (1, d)
        try:
            X1 = np.random.rand(1, d)
            Y1 = self.evaluate(X1)
            results["single_sample_2d"] = Y1.shape == (1, m)
        except Exception as e:
            results["single_sample_2d"] = False
            results["single_sample_2d_err"] = str(e)

        # 2. Single sample (d,)
        try:
            X1d = np.random.rand(d)
            Y1d = self.evaluate(X1d)
            results["single_sample_1d"] = Y1d.shape == (1, m)
        except Exception:
            results["single_sample_1d"] = False

        # 3. Batch sample
        try:
            X10 = np.random.rand(10, d)
            Y10 = self.evaluate(X10)
            results["batch_sample"] = Y10.shape == (10, m)
        except Exception:
            results["batch_sample"] = False

        # 4. Normalisation round-trip
        try:
            X_norm = np.random.rand(5, d)
            X_phys = self.denormalize(X_norm)
            X_norm2 = self.normalize(X_phys)
            results["normalization"] = np.allclose(X_norm, X_norm2, rtol=1e-8)
        except Exception:
            results["normalization"] = False

        # 5. Reproducibility
        try:
            X5 = np.random.rand(5, d)
            Y_a = self.evaluate(X5)
            Y_b = self.evaluate(X5)
            results["reproducible"] = np.array_equal(Y_a, Y_b)
        except Exception:
            results["reproducible"] = False

        # 6. No NaN in normal range
        try:
            X_lhs = self.sample(50, seed=0)
            Y_lhs = self.evaluate(X_lhs)
            n_nan = int(np.sum(np.isnan(Y_lhs)))
            results["no_nan"] = n_nan == 0
            if n_nan > 0:
                results["nan_count"] = n_nan
        except Exception:
            results["no_nan"] = False

        # 7. Output dtype
        try:
            Y_dtype = self.evaluate(np.random.rand(3, d))
            results["output_dtype_float64"] = Y_dtype.dtype == np.float64
        except Exception:
            results["output_dtype_float64"] = False

        # 8. Timing
        try:
            X_t = np.random.rand(20, d)
            t0 = time.perf_counter()
            self.evaluate(X_t)
            elapsed_ms = (time.perf_counter() - t0) * 1000 / 20
            limit = max(self.meta.typical_eval_time_ms.get("default", 1000.0) * 10, 5.0)
            results["timing"] = elapsed_ms < limit
            results["timing_ms_per_sample"] = round(elapsed_ms, 2)
        except Exception:
            results["timing"] = False

        passed = sum(1 for k, v in results.items() if isinstance(v, bool) and v)
        total = sum(1 for v in results.values() if isinstance(v, bool))
        results["summary"] = f"{passed}/{total} passed"
        return results

    # ── Convenience properties ────────────────────────────────────────────

    @property
    def dim(self) -> int:
        """Shortcut for input dimension."""
        return self.meta.dim_input

    @property
    def n_outputs(self) -> int:
        """Shortcut for output dimension."""
        return self.meta.dim_output

    def __repr__(self) -> str:
        m = self.meta
        return (
            f"<{self.__class__.__name__} "
            f"id='{m.case_id}' "
            f"dim={m.dim_input}→{m.dim_output} "
            f"solver='{m.solver_library}' "
            f"cost='{m.cost_level}'>"
        )
