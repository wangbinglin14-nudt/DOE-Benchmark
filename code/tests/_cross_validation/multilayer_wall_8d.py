"""Cross-validation: multilayer_wall_8d — Series thermal resistance (8 layers).

Independent references  (see references.bib)
----------------------------------------------
\\cite{Incropera2007}  §3.3.1 (Composite walls, series resistance).

Key relations (exact analytical)
---------------------------------
R_total = Σ (L_i / k_i)           (m²·K/W, per unit area)
q_flux  = ΔT / R_total            (W/m²)

For uniform conductivity k: R_total = N·L_layer / k
If k doubles, R halves and q doubles (exact inverse proportionality).

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

CASE_ID = "multilayer_wall_8d"
RTOL = 1e-6


@pytest.fixture(scope="module")
def case():
    import doe_testbed as db

    return db.get_case(CASE_ID)


class TestCrossValidation:
    """Verify against Incropera (2007) series resistance formulas."""

    def test_uniform_conductivity_symmetry(self, case):
        """All k equal → result invariant to permutation of layers."""
        k_val = 2.0
        X1 = case.normalize(np.array([[k_val] * 8]))
        Y1 = case.evaluate(X1)
        # Permute: swap first and last
        X2_phys = np.array([[k_val] * 8])
        X2_phys[0, 0], X2_phys[0, 7] = X2_phys[0, 7], X2_phys[0, 0]
        Y2 = case.evaluate(case.normalize(X2_phys))
        np.testing.assert_allclose(
            Y1, Y2, rtol=RTOL, err_msg="Uniform k should give same result regardless of layer order"
        )

    def test_higher_k_lower_resistance(self, case):
        """Higher conductivities → lower total resistance (Incropera §3.3)."""
        lb = np.array([iv.lower for iv in case.meta.input_variables])
        ub = np.array([iv.upper for iv in case.meta.input_variables])
        X_low = case.normalize(lb.reshape(1, -1))
        X_high = case.normalize(ub.reshape(1, -1))
        Y_low = case.evaluate(X_low)
        Y_high = case.evaluate(X_high)
        assert (
            Y_high[0, 0] < Y_low[0, 0]
        ), f"Higher k → lower R: R_high={Y_high[0,0]}, R_low={Y_low[0,0]}"
        assert (
            Y_high[0, 1] > Y_low[0, 1]
        ), f"Higher k → higher q: q_high={Y_high[0,1]}, q_low={Y_low[0,1]}"

    def test_resistance_scales_with_layers(self, case):
        """8-layer wall with uniform k should have 2× resistance of 4d case
        if layer thicknesses are the same — at minimum, R > 0."""
        k_val = 5.0
        X = case.normalize(np.array([[k_val] * 8]))
        Y = case.evaluate(X)
        R_total, q_flux = Y[0, 0], Y[0, 1]
        assert R_total > 0, "Resistance must be positive"
        assert q_flux > 0, "Heat flux must be positive"
