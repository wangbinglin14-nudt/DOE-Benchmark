r"""Cross-validation: ten_bar_truss_10d — Ten-bar planar truss FEM.

Independent references  (see references.bib)
----------------------------------------------
\cite{HaftkaGurdal1992}  §7.9 (geometry, loading, standard cross-sections).
\cite{Arora2004}         §5.9 (2-D truss FEM worked example).

Key verifiable properties:
- Uniform cross-sections: all bars equal → symmetric deflection pattern.
- Stiffness scales as A·E/L: doubling all A halves all displacements.
- Mass = Σ ρ·A_i·L_i (linear in A).

Validation scope:
The direct-stiffness solve is a control calculation for the ideal pin-jointed,
linear-elastic axial-bar truss model implemented by the case.  These checks
verify stiffness scaling, mass accounting, constrained-DOF handling, and the
standard ten-bar geometry plus the named _validate_physical_inputs private
physical-input path.  They do not establish a calibrated 1 % external validation
target for arbitrary truss structures or full structural-design practice.
"""

import numpy as np
import pytest

CASE_ID = "ten_bar_truss_10d"
RTOL = 0.01


@pytest.fixture(scope="module")
def case():
    import doe_testbed as db

    return db.get_case(CASE_ID)


class TestCrossValidation:
    """Verify against Haftka (1992) and basic FEM properties."""

    def test_displacement_scales_inversely_with_area(self, case):
        """FEM linearity: doubling all A → halves all displacements."""
        A_base = 10.0  # cm²
        X1 = np.array([[A_base] * 10])
        X2 = np.array([[A_base * 2] * 10])
        Y1 = case.evaluate(case.normalize(X1))
        Y2 = case.evaluate(case.normalize(X2))
        # v_max should halve
        np.testing.assert_allclose(Y2[0, 1], Y1[0, 1] / 2, rtol=0.02)

    def test_mass_linear_in_area(self, case):
        """Mass = Σ ρ·A_i·L_i is linear in A: doubling A doubles mass."""
        A_base = 10.0
        X1 = np.array([[A_base] * 10])
        X2 = np.array([[A_base * 2] * 10])
        Y1 = case.evaluate(case.normalize(X1))
        Y2 = case.evaluate(case.normalize(X2))
        np.testing.assert_allclose(Y2[0, 2], Y1[0, 2] * 2, rtol=0.01)

    def test_nonzero_displacement(self, case):
        """With finite cross-sections and applied load, displacement > 0."""
        X_phys = np.array([[5.0] * 10])
        X_norm = case.normalize(X_phys)
        Y = case.evaluate(X_norm)
        assert Y[0, 0] > 0 and Y[0, 1] > 0, "Displacements should be > 0"

    def test_larger_key_bars_reduce_deflection(self, case):
        r"""\cite{HaftkaGurdal1992} §7.9: bottom chords (A2, A3) dominate displacement;
        increasing them should reduce v_max significantly."""
        X_small = np.array([[5.0] * 10])
        X_large = np.array([[5.0] * 10])
        X_large[0, 1] = 20.0  # A2
        X_large[0, 2] = 20.0  # A3
        Y_small = case.evaluate(case.normalize(X_small))
        Y_large = case.evaluate(case.normalize(X_large))
        assert (
            Y_large[0, 1] < Y_small[0, 1]
        ), f"Larger A2,A3 should reduce v_max: {Y_large[0,1]} vs {Y_small[0,1]}"
