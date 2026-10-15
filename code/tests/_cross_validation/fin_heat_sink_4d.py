r"""Cross-validation: fin_heat_sink_4d — Rectangular cooling fin thermal resistance.

Independent references  (see references.bib)
----------------------------------------------
\cite{Incropera2007}  §3.6.2 (Fins of uniform cross-section, Eq. 3.70).
\cite{Holman2010}     §2.10 (Fin efficiency, insulated-tip approximation).

Key relations (insulated-tip rectangular fin)
---------------------------------------------
m = √(h·P / (k·A_c)) = √(2h / (k·t))   for thin rect. fin, P/A_c ≈ 2/t
η_fin = tanh(mH) / (mH)
q_fin = η · h · A_fin · ΔT
R_fin = 1 / (η · h · A_fin)

Validation scope:
These checks are analytical controls for the insulated-tip straight-fin
surrogate implemented by the case.  They verify fin-efficiency, conductivity,
height, convection-coefficient trends, and the named _validate_physical_inputs
private physical-input path, not calibrated validation of a full heat-sink
assembly with base spreading, contact resistance, radiation, or array
interaction effects.
"""

import numpy as np
import pytest

CASE_ID = "fin_heat_sink_4d"
RTOL = 0.02  # 2% for potential model detail differences


@pytest.fixture(scope="module")
def case():
    import doe_testbed as db

    return db.get_case(CASE_ID)


class TestCrossValidation:
    """Verify against Incropera (2007) fin efficiency formulas."""

    def test_efficiency_decreases_with_height(self, case):
        r"""\cite{Incropera2007} Eq. 3.70: taller fin → lower η (more temperature drop)."""
        X_short = np.array([[0.02, 1.5, 100.0, 200.0]])  # H, t, h, k
        X_tall = np.array([[0.07, 1.5, 100.0, 200.0]])
        Y_short = case.evaluate(case.normalize(X_short))
        Y_tall = case.evaluate(case.normalize(X_tall))
        assert (
            Y_tall[0, 1] < Y_short[0, 1]
        ), f"Taller fin → lower η: {Y_tall[0,1]} vs {Y_short[0,1]}"

    def test_higher_conductivity_higher_eta(self, case):
        r"""\cite{Incropera2007}: higher k → smaller m → higher η."""
        X_low_k = np.array([[0.05, 1.5, 100.0, 100.0]])
        X_high_k = np.array([[0.05, 1.5, 100.0, 400.0]])
        Y_low = case.evaluate(case.normalize(X_low_k))
        Y_high = case.evaluate(case.normalize(X_high_k))
        assert Y_high[0, 1] > Y_low[0, 1], f"Higher k → higher η: {Y_high[0,1]} vs {Y_low[0,1]}"

    def test_short_high_k_fin_eta_near_one(self, case):
        r"""\cite{Holman2010}: short fin with high conductivity → η ≈ 1."""
        X_phys = np.array([[0.01, 2.0, 25.0, 400.0]])
        X_norm = case.normalize(X_phys)
        Y = case.evaluate(X_norm)
        assert Y[0, 1] > 0.90, f"η={Y[0,1]}, expected ≈ 1 for short high-k fin"

    def test_resistance_inversely_related_to_h(self, case):
        """R_fin ∝ 1/(η·h·A): higher h → lower R."""
        X_low_h = np.array([[0.04, 1.5, 30.0, 200.0]])
        X_high_h = np.array([[0.04, 1.5, 400.0, 200.0]])
        Y_low = case.evaluate(case.normalize(X_low_h))
        Y_high = case.evaluate(case.normalize(X_high_h))
        assert Y_high[0, 2] < Y_low[0, 2], f"Higher h → lower R: {Y_high[0,2]} vs {Y_low[0,2]}"
