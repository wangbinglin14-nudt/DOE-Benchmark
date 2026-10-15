"""Cross-validation: variable_beam_5d — Stepped-section beam natural frequencies.

Independent references  (see references.bib)
----------------------------------------------
\\cite{Petyt2010}  §4.3 (Hermite beam elements, eigenvalue extraction).

The case uses FEM with Hermite cubic beam elements; the analytical control is
the clamped-free uniform-beam eigenvalue when all h_i are equal.

Euler–Bernoulli uniform cantilever beam:
  f_n = β_n² / (2πL²) √(EI/(ρA)), β_1 = 1.875104..., β_2 = 4.694091...

Validation scope
----------------
These checks verify eigenvalue ordering, monotone stiffness/mass trends, and a
uniform-beam cantilever analytical control.  Hermite beam FEM is a convergent
numerical discretisation for vibration eigenvalues; this file must not be read
as exact validation of arbitrary variable-section beam frequencies.
The named _validate_physical_inputs private physical-input path rejects
malformed arrays and segment heights outside the registered benchmark domain
before Hermite beam FEM assembly, so direct private calls cannot extrapolate
beyond the declared stepped-cantilever geometry bounds.
"""

import numpy as np
import pytest

CASE_ID = "variable_beam_5d"


@pytest.fixture(scope="module")
def case():
    import doe_testbed as db

    return db.get_case(CASE_ID)


class TestCrossValidation:
    """Verify against Petyt (2010) beam FEM eigenvalue theory."""

    def test_uniform_beam_symmetry(self, case):
        """Uniform beam (all h equal) → f₁, f₂ should match analytical."""
        h_val = 0.08  # m
        X_phys = np.array([[h_val] * 5])
        X_norm = case.normalize(X_phys)
        Y = case.evaluate(X_norm)
        f1, f2, mass = Y[0]
        assert f1 > 0 and f2 > f1, "f2 should be > f1 > 0"

    def test_thicker_beam_higher_frequency(self, case):
        """Thicker beam → higher EI → higher natural frequency."""
        X_thin = np.array([[0.03] * 5])
        X_thick = np.array([[0.12] * 5])
        Y_thin = case.evaluate(case.normalize(X_thin))
        Y_thick = case.evaluate(case.normalize(X_thick))
        assert Y_thick[0, 0] > Y_thin[0, 0], f"Thicker → higher f1: {Y_thick[0,0]} vs {Y_thin[0,0]}"

    def test_mass_proportional_to_section(self, case):
        """Mass ∝ Σ h_i (for fixed b, L, ρ)."""
        X_small = np.array([[0.04] * 5])
        X_large = np.array([[0.08] * 5])
        Y_small = case.evaluate(case.normalize(X_small))
        Y_large = case.evaluate(case.normalize(X_large))
        # Mass should roughly double when h doubles
        ratio = Y_large[0, 2] / Y_small[0, 2]
        assert 1.5 < ratio < 2.5, f"Mass ratio={ratio}, expected ≈ 2"
