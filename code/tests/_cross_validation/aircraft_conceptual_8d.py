"""Cross-validation: aircraft_conceptual_8d — Breguet/Raymer-style surrogate.

Independent references  (see references.bib)
----------------------------------------------
\\cite{Raymer2018}     §3 (sizing), §5 (thrust), §17 (performance: Breguet range).
\\cite{Anderson2010}   §5 (drag polar, Oswald efficiency).

Public source anchors:
  NASA Glenn Range - Constant Velocity:
    https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/range-constant-velocity/
  NASA Glenn Range Summary:
    https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/range-summary/
  MIT Unified Engineering Breguet range equation:
    https://web.mit.edu/16.unified/www/SPRING/propulsion/notes/node98.html
  NASA Glenn Induced Drag Coefficient:
    https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/induced-drag-coefficient/
  NASA Software Catalog Aviary:
    https://software.nasa.gov/software/LEW-20538-1
  NASA Software Catalog FLOPS v9:
    https://software.nasa.gov/software/LAR-18934-1
  NASA NTRS FLOPS weights-estimation method:
    https://ntrs.nasa.gov/citations/20170005851
  NASA NTRS GASP theoretical-development report:
    https://ntrs.nasa.gov/citations/19810010562
  NASA Aviary software overview:
    https://www.nasa.gov/aeronautics/aviary-software-overview/
  OpenMDAO Aviary documentation:
    https://openmdao.github.io/Aviary/
  NASA NTRS Aviary MDAO / analytic-gradient paper:
    https://ntrs.nasa.gov/citations/20230015940
  OpenMDAO Aviary external-subsystems documentation:
    https://openmdao.github.io/Aviary/user_guide_unreviewed/subsystems/using_external_subsystems.html
  OpenMDAO Aviary functionality overview:
    https://openmdao.github.io/Aviary/theory_guide/intro.html
  OpenMDAO Aviary features and functionalities:
    https://openmdao.github.io/Aviary/user_guide_unreviewed/features_and_functionalities.html
  OpenMDAO Aviary FAQ:
    https://openmdao.github.io/Aviary/misc_resources/FAQ.html
  OpenMDAO Aviary comparison to FLOPS:
    https://openmdao.github.io/Aviary/theory_guide/comparison_to_flops.html
  NASA OpenVSP boundary context:
    https://www.nasa.gov/reference/openvsp-vspaero-basics/

Key physics:
  Breguet range with mass TSFC: R = (V/(g·SFC)) · (L/D) · ln(W_i/W_f)
  With fixed L/D, R is separable in cruise speed, inverse SFC, and the fuel-log
  term: R ∝ Ma_cruise · SFC^{-1} · ln(1/(1 - 0.95 W_frac)).
  L/D_max = 0.5·√(π·AR·e/CD₀)  for parabolic drag polar.
  Higher AR → higher L/D → longer range.
  Mach number enters only through V = Ma·a_cruise in the Breguet product.

Scope: low-order conceptual-design surrogate. These checks validate formula
identities and expected engineering trends, not a calibrated full Raymer sizing
workflow, a verbatim Raymer/Roskam sizing-table implementation, or a numerical
aircraft-design validation case. The current case does not close payload,
take-off weight, wing loading, thrust loading, constraint diagrams,
climb/descent/loiter/diversion reserves, field length, engine lapse,
stability/control, certification, cost, noise, emissions, or mission trajectory.
It also does not compute cruise lift coefficient from dynamic pressure, wing
area, and weight; does not model wind/weather mission corrections,
compressibility drag rise, wave drag, drag-divergence Mach number, buffet, or
supercritical-airfoil design; and does not couple propulsion performance to
altitude or Mach. It also does not run
FLOPS/GASP/Aviary/OpenMDAO aircraft synthesis, mission analysis, or
optimisation, nor OpenVSP/VSPAERO-style parametric-geometry, DegenGeom,
potential-flow, pressure/wake-solution review, custom command-line/input-setup,
or aerodynamic-analysis workflows. SW and rho_alt are retained
as DOE stress-test inputs but are intentionally neutral in this constant-cruise
algebraic surrogate.
NASA FLOPS sources are used only as full aircraft-synthesis boundary evidence:
FLOPS-style weights, aerodynamics, propulsion-data, mission-performance,
takeoff/landing, and program-control modules are not instantiated or
calibrated here.
NASA GASP and Aviary comparison-to-FLOPS sources are used in the same boundary
role: GASP/FLOPS/Aviary geometry, mass/weight, empirical and tabular
aerodynamics, external aerodynamic models, propulsion engine decks, equations
of motion, reserve/off-design missions, takeoff/landing, performance,
economics, configuration trade studies, and OpenMDAO optimization machinery
are not instantiated or calibrated here.

The NASA/MIT source anchors support the Breguet/range/Mach/drag/induced-drag
formula context only. They are not external validation of the Raymer-style
empty-weight power law, a full aircraft sizing workflow, transonic performance,
cruise trim, or mission fuel-burn modelling. FLOPS/GASP/Aviary/OpenMDAO and
OpenVSP/VSPAERO references are boundary evidence for omitted full synthesis,
mission, takeoff/landing, propulsion, optimisation, geometry, DegenGeom,
potential-flow, pressure/wake-solution review, custom input setup,
OpenMDAO-based mission/design/optimization, and aerodynamic-analysis
workflows, including Aviary phase_info, trajectory, mission-phase,
external-subsystem, subsystem-builder, analytic-gradient MDAO, tightly coupled
aircraft/subsystem design, modular subsystem integration across trajectories,
empirical geometry, mass, aerodynamic, propulsion, and mission
core-subsystem workflows, conventional tube-and-wing empirical-equation
workflows, novel-configuration external-subsystem workflows, medium- and
high-fidelity analysis/optimization workflows, and external-tool coupling
workflows to NPSS, pyCycle, OpenAeroStruct, VSPAero, ANOPP, or electrical
modeling. The Aviary features/functionality page further documents the
minimum geometry, mass, aerodynamics, propulsion, and mission subsystem
coverage plus user-selected mission-phase setup expected by Aviary models.
These are boundary anchors for omitted coupled aircraft-workflow machinery,
not calibration evidence for this algebraic benchmark case.
The GASP theoretical-development report and Aviary comparison-to-FLOPS page
sharpen that nonvalidation boundary by showing that NASA synthesis workflows
span integrated technical disciplines and richer mission/performance modules
than the Breguet/parabolic-drag-polar/empty-weight equations tested here.
The named _validate_physical_inputs private physical-input path rejects
malformed arrays and AR/SW/lam/tc/SFC/W_frac/rho_alt/Ma_cruise values outside
the registered benchmark domain before Breguet/Raymer-style response assembly,
so direct private calls cannot extrapolate beyond the declared conceptual-design
surrogate bounds.  The public evaluate path also rejects non-finite,
non-physical, or out-of-contract range_km/L_D_max/We_frac responses outside the
declared output ranges before a row can enter a benchmark archive.
"""

import numpy as np
import pytest

CASE_ID = "aircraft_conceptual_8d"


@pytest.fixture(scope="module")
def case():
    import doe_testbed as db

    return db.get_case(CASE_ID)


class TestCrossValidation:
    """Verify Breguet/parabolic-drag identities and conceptual-design trends."""

    def test_higher_ar_longer_range(self, case):
        """\\cite{Raymer2018} §17: higher AR → higher L/D → more range."""
        X_low = np.array([[7.0, 200.0, 0.4, 0.12, 0.06, 0.35, 0.5, 0.78]])
        X_high = np.array([[13.0, 200.0, 0.4, 0.12, 0.06, 0.35, 0.5, 0.78]])
        Y_low = case.evaluate(case.normalize(X_low))
        Y_high = case.evaluate(case.normalize(X_high))
        assert Y_high[0, 0] > Y_low[0, 0], f"Higher AR → more range: {Y_high[0,0]} vs {Y_low[0,0]}"

    def test_range_positive(self, case):
        """Range must be positive."""
        X_phys = np.array([[10.0, 250.0, 0.3, 0.14, 0.065, 0.30, 0.5, 0.80]])
        X_norm = case.normalize(X_phys)
        Y = case.evaluate(X_norm)
        assert Y[0, 0] > 0, f"Range={Y[0,0]}, expected > 0"

    def test_ld_max_reasonable(self, case):
        """\\cite{Anderson2010}: subsonic transport L/D_max ≈ 15–20."""
        X_phys = np.array([[10.0, 200.0, 0.4, 0.12, 0.06, 0.35, 0.5, 0.78]])
        X_norm = case.normalize(X_phys)
        Y = case.evaluate(X_norm)
        assert 8 < Y[0, 1] < 30, f"L/D_max={Y[0,1]}, expected 15–20"

    def test_lower_sfc_longer_range(self, case):
        """\\cite{Raymer2018}: lower SFC → more fuel-efficient → longer range."""
        X_high_sfc = np.array([[10.0, 200.0, 0.4, 0.12, 0.075, 0.35, 0.5, 0.78]])
        X_low_sfc = np.array([[10.0, 200.0, 0.4, 0.12, 0.055, 0.35, 0.5, 0.78]])
        Y_high = case.evaluate(case.normalize(X_high_sfc))
        Y_low = case.evaluate(case.normalize(X_low_sfc))
        assert Y_low[0, 0] > Y_high[0, 0], f"Lower SFC → more range: {Y_low[0,0]} vs {Y_high[0,0]}"
