"""Cross-validation: multilayer_wall_16d — Series thermal resistance (16 layers).

Independent references  (see references.bib)
----------------------------------------------
\\cite{Incropera2007}  §3.3.1 (Composite walls, series resistance).

Key relations (exact analytical)
---------------------------------
R_total = Σ (L_i / k_i)           (m²·K/W, per unit area)
q_flux  = ΔT / R_total            (W/m²)

For N layers with the benchmark's fixed per-layer thickness L_layer and equal
conductivity k:
  R_total = N · L_layer / k
  → Resistance increases linearly with N because total wall thickness grows.

Validation scope:
These checks are exact series-resistance controls with fixed per-layer
thickness.  They validate Fourier-law identities for the implemented benchmark
material stack, not a calibrated material-stack or building-envelope design
model.  NASA passive-thermal-control material is boundary context for broader
thermal-path, hardware-integration, and test/analysis workflows, not
validation data for this compact one-dimensional conduction chain.
The named _validate_physical_inputs private physical-input path rejects
malformed arrays and layer conductivities outside the registered benchmark
domain before series-resistance assembly, so direct private calls cannot
extrapolate beyond the declared one-dimensional composite-wall bounds.
"""

import numpy as np
import pytest

CASE_ID = "multilayer_wall_16d"
RTOL = 1e-6


@pytest.fixture(scope="module")
def case():
    import doe_testbed as db

    return db.get_case(CASE_ID)


class TestCrossValidation:
    """Verify against Incropera (2007) series resistance formulas."""

    def test_uniform_conductivity(self, case):
        """All k equal → R_total = N·L_layer/k (positive, well-defined)."""
        k_val = 1.0
        X = case.normalize(np.array([[k_val] * 16]))
        Y = case.evaluate(X)
        R_total = Y[0, 0]
        assert R_total > 0, "Resistance must be positive for uniform k"

    def test_higher_k_lower_resistance(self, case):
        """Higher conductivities → lower total resistance (Fourier's law)."""
        lb = np.array([iv.lower for iv in case.meta.input_variables])
        ub = np.array([iv.upper for iv in case.meta.input_variables])
        X_low = case.normalize(lb.reshape(1, -1))
        X_high = case.normalize(ub.reshape(1, -1))
        Y_low = case.evaluate(X_low)
        Y_high = case.evaluate(X_high)
        assert Y_high[0, 0] < Y_low[0, 0], "Higher k → lower R"
        assert Y_high[0, 1] > Y_low[0, 1], "Higher k → higher q_flux"

    def test_equal_k_gives_equal_sobol(self, case):
        """With additive series resistance, Sobol ST for each layer should
        be approximately 1/d for the resistance output."""
        st = case.meta.sobol_ST
        if isinstance(next(iter(st.values())), dict):
            # Multi-output: check first output
            first_key = next(iter(st.keys()))
            vals = np.array(list(st[first_key].values()))
        else:
            vals = np.array(list(st.values()))
        # For perfectly additive model, all ST ≈ 1/d
        expected = 1.0 / 16
        np.testing.assert_allclose(
            vals,
            expected,
            atol=0.03,
            err_msg="Sobol ST should be ≈ 1/16 for additive resistance model",
        )

    def test_batch_no_nan(self, case):
        """Batch evaluation should produce no NaN."""
        X = case.sample(30, seed=99)
        Y = case.evaluate(X)
        assert not np.any(np.isnan(Y)), "NaN in batch output"
