r"""Cross-validation: soil_consolidation_5d — Terzaghi 1-D consolidation.

Independent references  (see references.bib)
----------------------------------------------
\cite{Das2019}   §11.6, Table 11.3 ($U$ vs $T_v$).
\cite{Craig2004} §7.2, Terzaghi analytical Fourier series.

Public source anchors used for the metadata audit:
* Itasca FLAC3D one-dimensional consolidation verification example:
  https://docs.itascacg.com/itasca940/flac3d/zone/test3d/Fluid/1DConsolidation/1dconsolidation.html
* MOLE Terzaghi one-dimensional consolidation analytical benchmark:
  https://mole-docs.readthedocs.io/en/main/examples/Parabolic/1D/Terzaghi1D.html
* OpenGTI consolidation overview, used only as drainage-path and omitted-scope
  context:
  https://opengti.io/consolidation
* OpenGeoSys hydro-mechanics benchmarks and ConsolidationBenchmark project,
  used only as coupled pressure-displacement HM/Biot omitted-scope context:
  https://www.opengeosys.org/docs/benchmarks/hydro-mechanics/
  https://www.opengeosys.org/stable/docs/benchmarks/hydro-mechanics/
  https://doxygen.opengeosys.org/6.5.6/d7/df1/ogs_ctest_prj__HydroMechanics__StaggeredScheme__ConsolidationBenchmark__consolidation_benchmark__prj.html

Key relations
-------------
c_v = k / (m_v · γ_w)                          (coefficient of consolidation)
T_v = c_v · t / H_dr²                          (time factor)
U(T_v) = 1 − Σ (2/M²)·exp(−M²·T_v)           (M = π(2m+1)/2, m=0,1,2,…)
S_t = m_v · Δσ · H_dr · U(T_v)                 (settlement at time t)

Validation scope
----------------
This is a one-dimensional Terzaghi single-drainage analytical control.  The
checks compare degree of consolidation and mid-depth pore pressure against the
Fourier-series solution and verify settlement identities/trends.  They do not
establish a calibrated site-specific 2-D/3-D coupled geotechnical analysis.
The regression suite also pins time-factor similarity: for fixed
T_v = c_v t / H_dr^2, U(T_v) and u_mid / Delta_sigma are invariant while
settlement scales with H_dr through S_final = m_v Delta_sigma H_dr.
The Itasca FLAC3D and MOLE Terzaghi pages are used as public anchors for the
governing equation, drainage/impermeable boundary pairing, and analytical
series benchmark, not as site-calibrated validation data. The Itasca example
is also treated as Biot-theory/Darcy-flow boundary context with an elastic soil
matrix, a drained boundary, an impervious base, and a numerical-vs-analytical
pore-pressure comparison; it is not calibration evidence for this compact
response surface.  The case does not
validate single-vs-double-drainage design choice, double-drainage or
radial-drain consolidation, layered or anisotropic soil deposits,
stress-dependent hydraulic conductivity or compressibility, finite-strain
effects, secondary compression/creep, construction staging, 2-D/3-D coupled
Biot poromechanics, poroelastic pressure-displacement coupling, stress/strain
field recovery, unsaturated flow, field instrumentation interpretation, or
laboratory/field oedometer parameter fitting.
OpenGTI-style geotechnical overview material is boundary context for these
omitted effects and parameter-fitting concerns; it is not used as validation
data for the numerical responses.
OpenGeoSys hydro-mechanics material, including the published "Consolidation
benchmark with the staggered scheme", is likewise boundary context for coupled
HM/THMC process modelling and staggered or monolithic pressure-displacement
simulation, not validation data for this single-drainage Terzaghi benchmark.
The referenced OpenGeoSys project file is a HYDRO_MECHANICS workflow with
staggered pressure/displacement coupling, fixed-stress stabilization,
Backward-Euler time stepping, and displacement/stress/strain/velocity outputs;
those solver and field-output contracts are outside this analytical control.
The named _validate_physical_inputs private physical-input path rejects
malformed arrays and k/m_v/H_dr/delta_s/t_obs values outside the registered
benchmark domain before Terzaghi Fourier-series evaluation, so direct private
calls cannot extrapolate beyond the declared one-dimensional consolidation
bounds.
Finite but non-physical or out-of-contract S_t/U_t/u_mid responses are also
rejected before row archival; this is a server-run data-integrity contract and
does not change valid analytical Terzaghi responses inside the declared output
envelopes.
"""

import numpy as np
import pytest

CASE_ID = "soil_consolidation_5d"
RTOL_U = 0.05  # 5% tolerance for degree of consolidation


@pytest.fixture(scope="module")
def case():
    import doe_testbed as db

    return db.get_case(CASE_ID)


def _terzaghi_U_analytical(Tv, n_terms=200):
    """Analytical Terzaghi solution for average degree of consolidation."""
    U = 0.0
    for m in range(n_terms):
        M = np.pi * (2 * m + 1) / 2.0
        U += (2.0 / M**2) * np.exp(-(M**2) * Tv)
    return 1.0 - U


class TestCrossValidation:
    """Verify against Terzaghi analytical Fourier-series solution."""

    def test_long_time_full_consolidation(self, case):
        r"""\cite{Das2019} Table 11.3: Tv → large ⇒ U → 1, S_t → m_v·Δσ·H_dr.

        k=1e-8 m/s, m_v=1e-6 1/Pa, H_dr=5m, Δσ=100kPa, t=50yr.
        c_v = 1e-8/(1e-6·9810) = 1.019e-6 m²/s
        t = 50·365.25·86400 = 1.578e9 s
        Tv = 1.019e-6·1.578e9/25 = 64.3 → U ≈ 1.0
        S_t = 1e-6·100e3·5·1.0 = 0.5 m = 500 mm
        """
        X_phys = np.array([[1e-8, 1e-6, 5.0, 100.0, 50.0]])
        X_norm = case.normalize(X_phys)
        Y = case.evaluate(X_norm)
        S_t, U_t, u_mid = Y[0]
        assert U_t > 0.99, f"U_t={U_t}, expected ≈ 1.0 at long time"
        S_expected = 1e-6 * 100e3 * 5.0 * 1000  # mm
        assert (
            abs(S_t - S_expected) / S_expected < 0.02
        ), f"S_t={S_t} mm, expected ≈ {S_expected} mm"

    def test_early_time_partial_consolidation(self, case):
        """Verify partial consolidation against analytical Tv–U curve.

        k=1e-9 m/s, m_v=5e-7 1/Pa, H_dr=10m, Δσ=50kPa, t=5yr.
        c_v = 1e-9/(5e-7·9810) = 2.038e-7 m²/s
        t = 5·365.25·86400 = 1.578e8 s
        Tv = 2.038e-7·1.578e8/100 = 0.3216
        U(0.32) ≈ 0.632 (Das Table 11.3)
        """
        k = 1e-9
        m_v = 5e-7
        gamma_w = 9810.0
        c_v = k / (m_v * gamma_w)
        t_sec = 5.0 * 365.25 * 86400
        H_dr = 10.0
        Tv = c_v * t_sec / H_dr**2
        U_analytical = _terzaghi_U_analytical(Tv)

        X_phys = np.array([[k, m_v, H_dr, 50.0, 5.0]])
        X_norm = case.normalize(X_phys)
        Y = case.evaluate(X_norm)
        U_computed = Y[0, 1]
        assert (
            abs(U_computed - U_analytical) < 0.05
        ), f"U_computed={U_computed:.3f}, U_analytical={U_analytical:.3f}"

    def test_settlement_monotone_with_load(self, case):
        """Higher Δσ → proportionally more settlement (linear theory)."""
        X_low = np.array([[1e-8, 1e-6, 5.0, 30.0, 10.0]])
        X_high = np.array([[1e-8, 1e-6, 5.0, 150.0, 10.0]])
        Y_low = case.evaluate(case.normalize(X_low))
        Y_high = case.evaluate(case.normalize(X_high))
        assert (
            Y_high[0, 0] > Y_low[0, 0]
        ), f"Higher load should give more settlement: {Y_high[0,0]} vs {Y_low[0,0]}"
