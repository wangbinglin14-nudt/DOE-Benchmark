"""
benchmark.surrogates
====================
Surrogate model adapters with a uniform fit/predict interface.

Each surrogate accepts ``Y_train`` of shape ``(n, m)`` and predicts on
``X_test`` returning ``(mean, std)`` of shape ``(n_test, m)``.  Models
that do not provide native uncertainty estimates return ``std = None``.

Implementations
---------------
- :class:`GPSurrogate`         — Gaussian Process (sklearn, Matérn 5/2 + ARD)
- :class:`RBFSurrogate`        — Radial-basis-function interpolant (scipy)
- :class:`RandomForestSurrogate`— Random Forest (sklearn)
- :class:`MLPSurrogate`        — Multi-layer perceptron (sklearn)
- :class:`PolynomialSurrogate` — Polynomial regression (sklearn pipeline)
- :class:`LinearSurrogate`     — Plain ordinary least squares (baseline)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from benchmark.core import Surrogate


# ─── Helper: per-output model wrapper ────────────────────────────────────
class _PerOutputModel:
    """Train one sklearn-style model per output column."""

    def __init__(self, factory):
        self._factory = factory
        self._models: List[Any] = []

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self._models = []
        for j in range(Y.shape[1]):
            m = self._factory()
            m.fit(X, Y[:, j])
            self._models.append(m)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.stack([m.predict(X) for m in self._models], axis=1)

    def predict_with_std(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        means: List[np.ndarray] = []
        stds: List[np.ndarray] = []
        for m in self._models:
            mu, sd = m.predict(X, return_std=True)
            means.append(mu)
            stds.append(sd)
        return np.stack(means, axis=1), np.stack(stds, axis=1)


# ─── Gaussian Process ────────────────────────────────────────────────────
class GPSurrogate(Surrogate):
    """Gaussian Process with Matérn 5/2 + ARD; sklearn implementation."""

    name = "gp"

    def __init__(self, length_scale_init: float = 1.0, n_restarts: int = 3) -> None:
        self.length_scale_init = float(length_scale_init)
        self.n_restarts = int(n_restarts)
        self._model: Optional[_PerOutputModel] = None
        self._y_mean: Optional[np.ndarray] = None
        self._y_std: Optional[np.ndarray] = None

    def get_init_kwargs(self) -> Dict[str, Any]:
        return {
            "length_scale_init": self.length_scale_init,
            "n_restarts": self.n_restarts,
        }

    def _make_model(self, d: int):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import (
            ConstantKernel,
            Matern,
            WhiteKernel,
        )

        # ARD: separate length_scale per dimension.
        # Bounds widened from the historical (1e-3, 1e3) to (1e-2, 1e4)
        # because real DOE responses regularly hit the upper bound at
        # high d when the response is nearly linear in some
        # coordinates; sklearn then emits a "ConvergenceWarning:
        # close to upper bound" for every restart, wasting CPU.
        # The white-noise lower bound is dropped to 1e-12 for the same
        # reason on the noise side.
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(
                length_scale=np.ones(d) * self.length_scale_init,
                length_scale_bounds=(1e-2, 1e4),
                nu=2.5,
            )
            + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-12, 1.0))
        )
        return GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=self.n_restarts,
            normalize_y=False,        # we standardise externally
            random_state=0,
            # Tikhonov regulariser added to the kernel matrix diagonal
            # before Cholesky.  1e-8 keeps factorisation stable on
            # near-coplanar designs without measurably biasing the
            # posterior.
            alpha=1e-8,
        )

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> "GPSurrogate":
        d = X_train.shape[1]
        # Standardise outputs for numerical stability (GPs are sensitive)
        self._y_mean = Y_train.mean(axis=0)
        self._y_std = Y_train.std(axis=0, ddof=1)
        self._y_std = np.where(self._y_std > 1e-12, self._y_std, 1.0)
        Y_std = (Y_train - self._y_mean) / self._y_std

        self._model = _PerOutputModel(lambda: self._make_model(d))
        self._model.fit(X_train, Y_std)
        return self

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._model is None:
            raise RuntimeError("call fit() before predict()")
        mu_std, sd_std = self._model.predict_with_std(X_test)
        # De-standardise
        mu = mu_std * self._y_std + self._y_mean
        sd = sd_std * self._y_std
        return mu, sd


# ─── RBF interpolant ─────────────────────────────────────────────────────
class RBFSurrogate(Surrogate):
    """Radial-basis-function interpolant (scipy.interpolate.RBFInterpolator)."""

    name = "rbf"

    def __init__(self, kernel: str = "thin_plate_spline", smoothing: float = 0.0) -> None:
        self.kernel = kernel
        self.smoothing = float(smoothing)
        self._models: List[Any] = []

    def get_init_kwargs(self) -> Dict[str, Any]:
        return {"kernel": self.kernel, "smoothing": self.smoothing}

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> "RBFSurrogate":
        from scipy.interpolate import RBFInterpolator

        self._models = []
        for j in range(Y_train.shape[1]):
            self._models.append(
                RBFInterpolator(
                    X_train, Y_train[:, j],
                    kernel=self.kernel, smoothing=self.smoothing,
                )
            )
        return self

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        mean = np.stack([m(X_test) for m in self._models], axis=1)
        return mean, None


# ─── Random Forest ───────────────────────────────────────────────────────
class RandomForestSurrogate(Surrogate):
    """Random Forest regression; std from per-tree predictions."""

    name = "rf"

    def __init__(self, n_estimators: int = 200, max_depth: Optional[int] = None) -> None:
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self._models: List[Any] = []

    def get_init_kwargs(self) -> Dict[str, Any]:
        return {"n_estimators": self.n_estimators, "max_depth": self.max_depth}

    def _make(self):
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=0,
            n_jobs=1,
        )

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> "RandomForestSurrogate":
        self._models = []
        for j in range(Y_train.shape[1]):
            m = self._make()
            m.fit(X_train, Y_train[:, j])
            self._models.append(m)
        return self

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        means: List[np.ndarray] = []
        stds: List[np.ndarray] = []
        for m in self._models:
            preds_per_tree = np.stack(
                [tree.predict(X_test) for tree in m.estimators_], axis=0
            )  # (n_trees, n_test)
            means.append(preds_per_tree.mean(axis=0))
            stds.append(preds_per_tree.std(axis=0, ddof=1))
        return np.stack(means, axis=1), np.stack(stds, axis=1)


# ─── MLP ─────────────────────────────────────────────────────────────────
class MLPSurrogate(Surrogate):
    """Multi-layer perceptron; deterministic given seed.  No uncertainty."""

    name = "mlp"

    def __init__(
        self,
        hidden_layer_sizes: tuple = (64, 64),
        max_iter: int = 2000,
        alpha: float = 1e-4,
    ) -> None:
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.max_iter = int(max_iter)
        self.alpha = float(alpha)
        self._model: Optional[_PerOutputModel] = None

    def get_init_kwargs(self) -> Dict[str, Any]:
        return {
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "max_iter": self.max_iter,
            "alpha": self.alpha,
        }

    def _make(self):
        from sklearn.neural_network import MLPRegressor
        return MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iter=self.max_iter,
            alpha=self.alpha,
            random_state=0,
            early_stopping=False,
        )

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> "MLPSurrogate":
        self._model = _PerOutputModel(self._make)
        self._model.fit(X_train, Y_train)
        return self

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return self._model.predict(X_test), None


# ─── Polynomial regression ───────────────────────────────────────────────
class PolynomialSurrogate(Surrogate):
    """Polynomial regression with Ridge regularisation."""

    name = "poly"

    def __init__(self, degree: int = 3, alpha: float = 1e-3) -> None:
        self.degree = int(degree)
        self.alpha = float(alpha)
        self._model: Optional[_PerOutputModel] = None

    def get_init_kwargs(self) -> Dict[str, Any]:
        return {"degree": self.degree, "alpha": self.alpha}

    def _make(self):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import PolynomialFeatures
        return Pipeline([
            ("poly", PolynomialFeatures(degree=self.degree, include_bias=False)),
            ("ridge", Ridge(alpha=self.alpha)),
        ])

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> "PolynomialSurrogate":
        self._model = _PerOutputModel(self._make)
        self._model.fit(X_train, Y_train)
        return self

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return self._model.predict(X_test), None


# ─── Linear (OLS) ────────────────────────────────────────────────────────
class LinearSurrogate(Surrogate):
    """Ordinary least squares (low-bias baseline)."""

    name = "linear"

    def __init__(self) -> None:
        self._model: Optional[_PerOutputModel] = None

    def get_init_kwargs(self) -> Dict[str, Any]:
        return {}

    def _make(self):
        from sklearn.linear_model import LinearRegression
        return LinearRegression()

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> "LinearSurrogate":
        self._model = _PerOutputModel(self._make)
        self._model.fit(X_train, Y_train)
        return self

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return self._model.predict(X_test), None


# ─── SVR (Support Vector Regression) ─────────────────────────────────────
class SVRSurrogate(Surrogate):
    """Support Vector Regression with RBF kernel.

    A common alternative to GP in DOE / surrogate-modelling literature.
    Does not provide native uncertainty estimates.
    """

    name = "svr"

    def __init__(self, C: float = 1.0, epsilon: float = 0.05, gamma: str = "scale") -> None:
        self.C = float(C)
        self.epsilon = float(epsilon)
        self.gamma = gamma
        self._model: Optional[_PerOutputModel] = None
        self._y_mean: Optional[np.ndarray] = None
        self._y_std: Optional[np.ndarray] = None

    def get_init_kwargs(self) -> Dict[str, Any]:
        return {"C": self.C, "epsilon": self.epsilon, "gamma": self.gamma}

    def _make(self):
        from sklearn.svm import SVR
        return SVR(C=self.C, epsilon=self.epsilon, gamma=self.gamma, kernel="rbf")

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> "SVRSurrogate":
        # SVR is sensitive to output scale — standardise per output column
        self._y_mean = Y_train.mean(axis=0)
        self._y_std = Y_train.std(axis=0, ddof=1)
        self._y_std = np.where(self._y_std > 1e-12, self._y_std, 1.0)
        Y_std = (Y_train - self._y_mean) / self._y_std
        self._model = _PerOutputModel(self._make)
        self._model.fit(X_train, Y_std)
        return self

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        Y_std = self._model.predict(X_test)
        Y = Y_std * self._y_std + self._y_mean
        return Y, None


# ─── KNN (k-nearest-neighbours regression) ───────────────────────────────
class KNNSurrogate(Surrogate):
    """K-nearest-neighbours regression baseline.

    Std is estimated from the std of the k neighbours' values.
    """

    name = "knn"

    def __init__(self, n_neighbors: int = 5, weights: str = "distance") -> None:
        self.n_neighbors = int(n_neighbors)
        self.weights = weights
        self._model: Optional[_PerOutputModel] = None
        self._X_train: Optional[np.ndarray] = None
        self._Y_train: Optional[np.ndarray] = None

    def get_init_kwargs(self) -> Dict[str, Any]:
        return {"n_neighbors": self.n_neighbors, "weights": self.weights}

    def _make(self):
        from sklearn.neighbors import KNeighborsRegressor
        return KNeighborsRegressor(
            n_neighbors=min(self.n_neighbors, 50),
            weights=self.weights,
        )

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> "KNNSurrogate":
        self._X_train = X_train.copy()
        self._Y_train = Y_train.copy()
        # Adapt n_neighbors when training set is small
        k = max(1, min(self.n_neighbors, len(X_train) // 2))
        from sklearn.neighbors import KNeighborsRegressor
        self._model = _PerOutputModel(
            lambda: KNeighborsRegressor(n_neighbors=k, weights=self.weights)
        )
        self._model.fit(X_train, Y_train)
        return self

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        from sklearn.neighbors import NearestNeighbors
        # Mean prediction
        Y_mean = self._model.predict(X_test)
        # Std from neighbour spread
        k = max(2, min(self.n_neighbors, len(self._X_train) // 2))
        nn = NearestNeighbors(n_neighbors=k).fit(self._X_train)
        _, idx = nn.kneighbors(X_test)
        # idx shape (n_test, k)
        Y_neigh = self._Y_train[idx]   # (n_test, k, m)
        Y_std = np.std(Y_neigh, axis=1, ddof=1)
        return Y_mean, Y_std


# ─── Gradient Boosting ───────────────────────────────────────────────────
class GradientBoostingSurrogate(Surrogate):
    """Gradient Boosted Regression Trees (sklearn HistGradientBoosting).

    Modern alternative to Random Forest; often more accurate at the
    cost of less interpretability.  No native uncertainty.
    """

    name = "gb"

    def __init__(
        self,
        max_iter: int = 200,
        learning_rate: float = 0.1,
        max_depth: Optional[int] = 6,
    ) -> None:
        self.max_iter = int(max_iter)
        self.learning_rate = float(learning_rate)
        self.max_depth = max_depth
        self._model: Optional[_PerOutputModel] = None

    def get_init_kwargs(self) -> Dict[str, Any]:
        return {
            "max_iter": self.max_iter,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
        }

    def _make(self):
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=0,
        )

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> "GradientBoostingSurrogate":
        self._model = _PerOutputModel(self._make)
        self._model.fit(X_train, Y_train)
        return self

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return self._model.predict(X_test), None


# ─── GP with squared-exponential / RBF kernel (kernel ablation) ──────────
class GPRBFSurrogate(Surrogate):
    """Gaussian Process with RBF (squared exponential) kernel + ARD.

    Provides a kernel-ablation comparison against :class:`GPSurrogate`
    (which uses Matérn 5/2).  RBF assumes infinitely-differentiable
    sample paths — correct for analytic functions, often too smooth
    for discontinuous physical responses.
    """

    name = "gp_rbf"

    def __init__(self, length_scale_init: float = 1.0, n_restarts: int = 3) -> None:
        self.length_scale_init = float(length_scale_init)
        self.n_restarts = int(n_restarts)
        self._model: Optional[_PerOutputModel] = None
        self._y_mean: Optional[np.ndarray] = None
        self._y_std: Optional[np.ndarray] = None

    def get_init_kwargs(self) -> Dict[str, Any]:
        return {
            "length_scale_init": self.length_scale_init,
            "n_restarts": self.n_restarts,
        }

    def _make_model(self, d: int):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import (
            RBF,
            ConstantKernel,
            WhiteKernel,
        )
        # See `GPSurrogate._make_model` for the rationale on the
        # widened length-scale / noise bounds and the Tikhonov alpha.
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * RBF(
                length_scale=np.ones(d) * self.length_scale_init,
                length_scale_bounds=(1e-2, 1e4),
            )
            + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-12, 1.0))
        )
        return GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=self.n_restarts,
            normalize_y=False,
            random_state=0,
            alpha=1e-8,
        )

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> "GPRBFSurrogate":
        d = X_train.shape[1]
        self._y_mean = Y_train.mean(axis=0)
        self._y_std = Y_train.std(axis=0, ddof=1)
        self._y_std = np.where(self._y_std > 1e-12, self._y_std, 1.0)
        Y_std = (Y_train - self._y_mean) / self._y_std

        self._model = _PerOutputModel(lambda: self._make_model(d))
        self._model.fit(X_train, Y_std)
        return self

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._model is None:
            raise RuntimeError("call fit() before predict()")
        mu_std, sd_std = self._model.predict_with_std(X_test)
        mu = mu_std * self._y_std + self._y_mean
        sd = sd_std * self._y_std
        return mu, sd


# ─── Higher-order polynomial (PCE-like) ──────────────────────────────────
class PolyChaosSurrogate(Surrogate):
    """Higher-order polynomial regression with stronger Ridge regularisation.

    Approximates a Polynomial Chaos Expansion (PCE) with truncated
    monomial basis.  Useful for cases where the response is genuinely
    polynomial-like (Sobol indices analytical) — would suffer on
    threshold/discontinuous responses.
    """

    name = "poly_chaos"

    def __init__(self, degree: int = 5, alpha: float = 1e-2) -> None:
        self.degree = int(degree)
        self.alpha = float(alpha)
        self._model: Optional[_PerOutputModel] = None

    def get_init_kwargs(self) -> Dict[str, Any]:
        return {"degree": self.degree, "alpha": self.alpha}

    def _make(self):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import PolynomialFeatures
        return Pipeline([
            ("poly", PolynomialFeatures(degree=self.degree, include_bias=False)),
            ("ridge", Ridge(alpha=self.alpha)),
        ])

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> "PolyChaosSurrogate":
        self._model = _PerOutputModel(self._make)
        self._model.fit(X_train, Y_train)
        return self

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return self._model.predict(X_test), None


# ─── Registry ────────────────────────────────────────────────────────────
SURROGATE_REGISTRY = {
    cls.name: cls for cls in [
        GPSurrogate, GPRBFSurrogate, RBFSurrogate, RandomForestSurrogate,
        GradientBoostingSurrogate, MLPSurrogate,
        PolynomialSurrogate, PolyChaosSurrogate,
        SVRSurrogate, KNNSurrogate, LinearSurrogate,
    ]
}


def get_surrogate(name: str, **kwargs) -> Surrogate:
    if name not in SURROGATE_REGISTRY:
        raise KeyError(
            f"Unknown surrogate {name!r}; available: {list(SURROGATE_REGISTRY)}"
        )
    return SURROGATE_REGISTRY[name](**kwargs)
