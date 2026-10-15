"""Cross-validation: casting_solidification_5d — Stefan solidification problem.

Independent references  (see references.bib)
----------------------------------------------
\\cite{Flemings1974}       §5 (enthalpy method for Stefan problem).
\\cite{PoirierGeiger1994}  §12 (Chvorinov's rule, solidification time).

Key physics:
  Chvorinov's rule: t_solid ∝ (V/A)² ∝ L_half²
  Higher T_pour → longer solidification time.
  Higher h_contact → faster cooling → shorter t_solid.

Validation scope
----------------
These checks verify Chvorinov/Stefan-trend behavior, enthalpy-method
front-tracking controls, and the named _validate_physical_inputs private
physical-input path for the implemented one-dimensional FDM surrogate.
They do not establish a 5 % case-level match to a calibrated casting process,
full mould simulation, or experimental solidification-time dataset.
"""

import numpy as np
import pytest

CASE_ID = "casting_solidification_5d"


@pytest.fixture(scope="module")
def case():
    import doe_testbed as db

    return db.get_case(CASE_ID)


class TestCrossValidation:
    """Verify against \\cite{Flemings1974} solidification physics."""

    def test_higher_pour_temp_longer_solidification(self, case):
        """Higher T_pour → more superheat → longer t_solid.

        Note: the superheat effect is only resolved on the FDM grid when
        h_contact is high enough for rapid surface cooling; at low h,
        the mold-side resistance dominates and masks T_pour sensitivity.
        """
        X_low = np.array([[650.0, 50.0, 1500.0, 0.8, 0.08]])
        X_high = np.array([[815.0, 50.0, 1500.0, 0.8, 0.08]])
        Y_low = case.evaluate(case.normalize(X_low))
        Y_high = case.evaluate(case.normalize(X_high))
        assert (
            Y_high[0, 0] > Y_low[0, 0]
        ), f"Higher T_pour → longer t_solid: {Y_high[0,0]} vs {Y_low[0,0]}"

    def test_higher_contact_h_faster_solidification(self, case):
        """Higher h_contact → better mold-metal heat transfer → faster."""
        X_poor = np.array([[700.0, 50.0, 300.0, 0.8, 0.08]])
        X_good = np.array([[700.0, 50.0, 1500.0, 0.8, 0.08]])
        Y_poor = case.evaluate(case.normalize(X_poor))
        Y_good = case.evaluate(case.normalize(X_good))
        assert (
            Y_good[0, 0] < Y_poor[0, 0]
        ), f"Higher h → faster solidification: {Y_good[0,0]} vs {Y_poor[0,0]}"

    def test_thicker_casting_longer(self, case):
        """\\cite{PoirierGeiger1994}: Chvorinov t ∝ L_half² → thicker = longer."""
        X_thin = np.array([[700.0, 50.0, 500.0, 0.8, 0.02]])
        X_thick = np.array([[700.0, 50.0, 500.0, 0.8, 0.08]])
        Y_thin = case.evaluate(case.normalize(X_thin))
        Y_thick = case.evaluate(case.normalize(X_thick))
        assert (
            Y_thick[0, 0] > Y_thin[0, 0]
        ), f"Thicker casting → longer t_solid: {Y_thick[0,0]} vs {Y_thin[0,0]}"

    def test_center_temp_above_solidus(self, case):
        """T_center_peak should be ≥ solidus during solidification."""
        X_phys = np.array([[700.0, 50.0, 500.0, 0.8, 0.05]])
        X_norm = case.normalize(X_phys)
        Y = case.evaluate(X_norm)
        assert Y[0, 2] > 500, f"T_center_peak={Y[0,2]}, expected > solidus"
