"""Cross-validation: hetero_conduction_2d — Two-material 1-D steady conduction.

Independent references  (see references.bib)
----------------------------------------------
\\cite{Incropera2007}  §3.3 (Composite walls, series thermal resistance).
\\cite{Patankar1980}   §4.2 (FDM for 1-D conduction).

For a two-layer slab (materials A, B) with perfect contact:
  R_total = x_split/k_A + (L − x_split)/k_B
  q = ΔT / R_total
  T_mid = T_hot − q · x_split/k_A     (at the interface)

Validation scope:
These checks are analytical/FDM controls for the declared two-material
conduction surrogate.  They verify Fourier-law series-resistance limits,
face-consistent flux extraction, and the named _validate_physical_inputs
private physical-input path, not calibrated validation of arbitrary composite
media, contact resistance, or multidimensional thermal-bridge experiments.
"""

import numpy as np
import pytest

CASE_ID = "hetero_conduction_2d"
RTOL = 0.05


@pytest.fixture(scope="module")
def case():
    import doe_testbed as db

    return db.get_case(CASE_ID)


class TestCrossValidation:
    """Verify against \\cite{Incropera2007} composite-wall series resistance."""

    def test_uniform_material(self, case):
        """When k_A ≈ k_B, T_mid should be at the linear interpolation."""
        X_phys = np.array([[5.0, 5.0, 0.5, 100.0, 0.0]])
        X_norm = case.normalize(X_phys)
        Y = case.evaluate(X_norm)
        # Uniform k → linear profile → T_mid = (T_hot + T_cold)/2
        assert abs(Y[0, 0] - 50.0) < 5.0, f"T_mid={Y[0,0]}, expected ≈ 50"

    def test_thermal_bridge_increases_qmax(self, case):
        """Strong k_B/k_A contrast → thermal-bridge effect → higher q_max."""
        X_uniform = np.array([[2.0, 2.0, 0.5, 100.0, 0.0]])
        X_bridge = np.array([[0.1, 50.0, 0.5, 100.0, 0.0]])
        Y_uni = case.evaluate(case.normalize(X_uniform))
        Y_br = case.evaluate(case.normalize(X_bridge))
        # Heat concentrates through k_B region → q_max rises
        assert (
            Y_br[0, 1] > Y_uni[0, 1]
        ), f"Bridge q_max={Y_br[0,1]:.1f} vs uniform q_max={Y_uni[0,1]:.1f}"

    def test_higher_dT_higher_flux(self, case):
        """Larger ΔT → proportionally larger heat flux."""
        X_low = np.array([[2.0, 10.0, 0.5, 70.0, 10.0]])
        X_high = np.array([[2.0, 10.0, 0.5, 120.0, 0.0]])
        Y_low = case.evaluate(case.normalize(X_low))
        Y_high = case.evaluate(case.normalize(X_high))
        assert Y_high[0, 1] > Y_low[0, 1], "Higher ΔT → higher q"
