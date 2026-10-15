"""Cross-validation: dam_seepage_6d — 2-D steady-state Darcy seepage uplift.

Independent references  (see references.bib)
----------------------------------------------
Bear, J. (1972). Dynamics of Fluids in Porous Media. Elsevier.
SL 319-2018 Design Code for Concrete Gravity Dams.

Public source anchors used for the metadata audit:
* USACE EM 1110-2-2200, Gravity Dam Design:
  https://www.publications.usace.army.mil/Portals/76/Publications/EngineerManuals/EM_1110-2-2200.pdf
* USACE EM 1110-2-1901, Seepage Analysis and Control for Dams:
  https://www.publications.usace.army.mil/Portals/76/Publications/EngineerManuals/EM_1110-2-1901.pdf
* Association of State Dam Safety Officials summary for USACE EM 1110-2-1901:
  https://damsafety.org/content/seepage-analysis-and-control-dams
* USACE EM 1110-2-1914, Design, Construction, and Maintenance of Relief Wells:
  https://www.publications.usace.army.mil/Portals/76/Publications/EngineerManuals/EM%201110-2-1914_FINAL%2020250307.pdf
  USACE Digital Library historical record:
  https://usace.contentdm.oclc.org/digital/collection/p16021coll9/id/2595/
* USACE ERDC/GSL TR-25-5, Relief Wells for Dams and Levees: History and
  Current Practice:
  https://www.erdc.usace.army.mil/Media/Publication-Notices/Article/4139695/usace-relief-wells-for-dams-and-levees-history-and-current-practice/
* USBR Design Standards No. 13, Chapter 8, Seepage:
  https://www.usbr.gov/tsc/techreferences/designstandards-datacollectionguides/finalds-pdfs/DS13-8.pdf
* FERC Engineering Guidelines, Chapter 3, Gravity Dams:
  https://www.ferc.gov/sites/default/files/2020-04/chap3.pdf
* FERC Risk-Informed Decision Making:
  https://www.ferc.gov/dam-safety-and-inspections/risk-informed-decision-making-ridm
* FERC Potential Failure Modes / DSPMP-PFMA pages:
  https://www.ferc.gov/dam-safety-and-inspections/potential-failure-modes-pfms
  https://www.ferc.gov/dam-safety-and-inspections/dam-safety-performance-monitoring-program-dspmp-and-potential-failure
* FEMA P-1032, Evaluation and Monitoring of Seepage and Internal Erosion:
  https://www.fema.gov/sites/default/files/2020-08/fema_p1032_eval_monitoring_seepage_internal_erosion.pdf

Key relations
-------------
Darcy seepage (Laplace): ∇·(K·∇H) = 0
Uplift: u(x) = γ_w · H(x, z=0)
U_total = ∫ u(x) dx  (total uplift force per unit width)
U_center = u(L_base/2) from the exact dam-base midpoint, not the nearest
grid node

Without curtain or drain, head varies linearly from H1 to H2:
  U_no_curtain = γ_w · (H1+H2)/2 · L_base
This matches the FERC gravity-dam review baseline that unverified drainage
should not be credited and uplift at the foundation contact is treated as a
straight-line headwater-to-tailwater distribution over the base area.

Validation scope:
These checks are plausibility and limiting-case controls for the implemented
Darcy/FDM uplift surrogate.  They verify monotone head/drain trends, positive
outputs, the full homogeneous no-drain linear base-head profile, and the
corresponding linear-uplift limit, not calibrated external validation of a
dam-foundation seepage design model.  The fixed-head drain parameter is a
benchmark pressure-relief surrogate, not a design-code drain-efficiency
recommendation and not a calibrated drain-hole-network parameter.  The relief
node activates only when its prescribed head is below the no-drain natural head
at the drain location, so the simplified drain cannot act as a recharge source
in strong-curtain cases.  The response is smooth within the inactive-drain and
active-drain regimes, with a threshold where the relief boundary switches on.
The case does not validate flow nets, piezometer
interpretation, permeability uncertainty studies, mesh convergence, relief
wells, relief-well design/construction/maintenance, filter design, internal
erosion, dam-safety monitoring/inspection programs, collector-pipe/header
layouts, relief-well spacing/sizing, rehabilitation planning,
seepage-control remediation selection, or structural stability.
The ASDSO/USACE seepage-control summary is used here only as boundary evidence
for omitted monitoring, inspection, maintenance/rehabilitation, filter, drain,
piping, and erosion-control workflows, not as validation of the single-node
drain surrogate.
Any real-world claim of uplift reduction would require instrumentation and
project-specific evidence; this benchmark does not provide that evidence.
USACE EM 1110-2-1901 and EM 1110-2-1914 therefore remain boundary material for
site-specific drains, galleries, collector pipes, relief wells, and
rehabilitation/maintenance workflows rather than validation of this compact
fixed-grid uplift surrogate.
The official relief-well manual also makes clear that real systems include
siting, spacing, hydraulic design, development/pump-testing, operation, and
maintenance workflows; none of those relief-system details are represented by
the benchmark's single pressure-relief node.
The 2025 USACE ERDC relief-well current-practice report reinforces the same
boundary for maintenance/repair practice, seepage-remediation use, and
pressure-relief system practice; it is not calibration evidence for the
registered uplift response.
FERC risk-informed and potential-failure-mode pages are likewise boundary
evidence: a real dam-safety evaluation considers loading likelihood, system
response, failure consequences, potential failure modes, performance monitoring,
surveillance, inspection, instrumentation, and risk-reduction measures.  This
three-output uplift surrogate does not estimate those quantities, does not
produce a seepage-velocity field or full pore-pressure field, and does not
define PFM/PFMA or risk-management decisions.
FEMA P-1032 reinforces the same omitted-workflow boundary for seepage and
internal erosion: real evaluations use visual observation, geophysical
investigation, seepage collection, flow-rate, pressure, and water-quality
monitoring, PFM identification, life-cycle monitoring, emergency response, and
remediation decisions.  This benchmark does not implement those monitoring or
internal-erosion workflows.
The named _validate_physical_inputs private physical-input path rejects malformed arrays and
H1/H2/K_base/K_curtain/D_base/drain_eff values outside the registered
benchmark domain before sparse FDM assembly, so direct private calls cannot
extrapolate beyond the declared uplift-surrogate bounds.
"""

import numpy as np
import pytest

CASE_ID = "dam_seepage_6d"


@pytest.fixture(scope="module")
def case():
    import doe_testbed as db

    return db.get_case(CASE_ID)


class TestCrossValidation:
    """Verify dam seepage uplift against Bear (1972) / SL 319-2018."""

    def test_higher_upstream_head_higher_uplift(self, case):
        """Higher H1 → higher total uplift force (Darcy linearity)."""
        # Low H1, moderate other params
        X_low = case.normalize(np.array([[30.0, 5.0, 1e-5, 1e-8, 30.0, 0.3]]))
        X_high = case.normalize(np.array([[80.0, 5.0, 1e-5, 1e-8, 30.0, 0.3]]))
        Y_low = case.evaluate(X_low)
        Y_high = case.evaluate(X_high)
        assert (
            Y_high[0, 0] > Y_low[0, 0]
        ), f"Higher H1 should give higher uplift: {Y_high[0,0]} vs {Y_low[0,0]}"

    def test_higher_drain_eff_reduces_uplift(self, case):
        """Higher drain efficiency → lower uplift (Bear 1972, §9.6).
        Compare two positive drain_eff values (both activate the Dirichlet
        drain BC); the transition from 0→nonzero changes BC type."""
        X_low_drain = case.normalize(np.array([[60.0, 5.0, 1e-5, 1e-7, 30.0, 0.2]]))
        X_high_drain = case.normalize(np.array([[60.0, 5.0, 1e-5, 1e-7, 30.0, 0.8]]))
        Y_low = case.evaluate(X_low_drain)
        Y_high = case.evaluate(X_high_drain)
        assert (
            Y_high[0, 0] < Y_low[0, 0]
        ), f"Higher drain_eff should reduce uplift: {Y_high[0,0]} vs {Y_low[0,0]}"

    def test_uplift_reduction_factor_bounded(self, case):
        """η_uplift should be in (0, 1] for this clipped reduction-factor surrogate."""
        X = case.sample(50, seed=42)
        Y = case.evaluate(X)
        eta = Y[:, 2]
        assert np.all(eta > 0) and np.all(
            eta <= 1.0
        ), f"eta range [{eta.min():.3f}, {eta.max():.3f}] outside (0, 1]"

    def test_outputs_positive(self, case):
        """All outputs should be non-negative for valid inputs."""
        X = case.sample(30, seed=7)
        Y = case.evaluate(X)
        assert np.all(Y >= 0), f"Negative output detected: min={Y.min()}"
