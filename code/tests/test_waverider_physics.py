"""Physics regression tests for ``waverider_4d``."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from doe_testbed.cases.aerodynamics.waverider_4d import Waverider4D


def _evaluate(case: Waverider4D, rows: list[list[float]]) -> np.ndarray:
    X = np.array(rows, dtype=float)
    return case.evaluate(case.normalize(X))


def test_modified_newtonian_cpmax_approaches_hypersonic_limit():
    cp = Waverider4D._cp_max(np.array([5.0, 12.0, 50.0]))

    assert np.all(cp > 0.0)
    assert cp[1] > cp[0]
    assert cp[2] > cp[1]
    assert cp[2] < 2.0


def test_modified_newtonian_cpmax_matches_gamma14_rayleigh_pitot_limit():
    gamma = 1.4
    high_mach = Waverider4D._cp_max(np.array([1000.0]), gamma=gamma)[0]
    asymptotic = (
        2.0
        / gamma
        * ((gamma + 1.0) ** 2 / (4.0 * gamma)) ** (gamma / (gamma - 1.0))
        * (2.0 * gamma / (gamma + 1.0))
    )

    assert np.isclose(high_mach, asymptotic, rtol=1e-6)
    assert np.isclose(asymptotic, 1.8393710511306678)


def test_rayleigh_pitot_cpmax_rejects_out_of_contract_inputs():
    with pytest.raises(ValueError, match="supersonic Mach"):
        Waverider4D._cp_max(np.array([1.0]))

    with pytest.raises(ValueError, match="supersonic Mach"):
        Waverider4D._cp_max(np.array([np.nan]))

    with pytest.raises(ValueError, match="invalid gamma"):
        Waverider4D._cp_max(np.array([8.0]), gamma=1.0)


def test_cone_angle_increases_lift_and_volume_efficiency():
    case = Waverider4D()

    shallow, steep = _evaluate(
        case,
        [
            [8.0, 5.0, 0.0, 60.0],
            [8.0, 20.0, 0.0, 60.0],
        ],
    )

    assert steep[1] > shallow[1]
    assert steep[2] > shallow[2]


def test_higher_mach_improves_ld_by_reducing_friction_drag():
    case = Waverider4D()
    Y = _evaluate(
        case,
        [
            [5.0, 10.0, 0.0, 60.0],
            [8.0, 10.0, 0.0, 60.0],
            [12.0, 10.0, 0.0, 60.0],
        ],
    )

    assert np.all(np.diff(Y[:, 0]) > 0.0)
    assert np.all(np.diff(Y[:, 1]) > 0.0)
    assert np.allclose(Y[:, 2], Y[0, 2], rtol=1e-12, atol=1e-12)


def test_angle_of_attack_increases_lift_but_reduces_ld():
    case = Waverider4D()
    Y = _evaluate(
        case,
        [
            [8.0, 10.0, 0.0, 60.0],
            [8.0, 10.0, 2.5, 60.0],
            [8.0, 10.0, 5.0, 60.0],
        ],
    )

    assert np.all(np.diff(Y[:, 1]) > 0.0)
    assert np.all(np.diff(Y[:, 0]) < 0.0)
    assert np.allclose(Y[:, 2], Y[0, 2], rtol=1e-12, atol=1e-12)


def test_span_angle_increases_aerodynamic_area_but_reduces_volume_efficiency():
    case = Waverider4D()
    Y = _evaluate(
        case,
        [
            [8.0, 10.0, 0.0, 30.0],
            [8.0, 10.0, 0.0, 60.0],
            [8.0, 10.0, 0.0, 90.0],
        ],
    )

    assert np.all(np.diff(Y[:, 0]) > 0.0)
    assert np.all(np.diff(Y[:, 1]) > 0.0)
    assert np.all(np.diff(Y[:, 2]) < 0.0)


def test_cone_angle_has_ld_tradeoff_while_volume_keeps_increasing():
    case = Waverider4D()
    Y = _evaluate(
        case,
        [
            [8.0, 5.0, 0.0, 60.0],
            [8.0, 10.0, 0.0, 60.0],
            [8.0, 20.0, 0.0, 60.0],
        ],
    )

    assert Y[1, 0] > Y[0, 0]
    assert Y[1, 0] > Y[2, 0]
    assert np.all(np.diff(Y[:, 1]) > 0.0)
    assert np.all(np.diff(Y[:, 2]) > 0.0)


def test_representative_hypersonic_ld_is_in_waverider_order_of_magnitude():
    case = Waverider4D()
    y = _evaluate(case, [[6.0, 10.0, 2.0, 60.0]])[0]

    assert 2.0 < y[0] < 8.0
    assert y[1] > 0.0
    assert 0.01 < y[2] < 0.15


def test_volume_efficiency_matches_geometric_similarity_formula():
    case = Waverider4D()
    X_phys = np.array(
        [
            [8.0, 5.0, 0.0, 30.0],
            [8.0, 10.0, 2.5, 60.0],
            [8.0, 20.0, 5.0, 90.0],
        ],
        dtype=float,
    )
    Y = case.evaluate(case.normalize(X_phys))
    theta_c = np.deg2rad(X_phys[:, 1])
    phi = np.deg2rad(X_phys[:, 3])
    expected_eta_v = 0.22 * np.tan(theta_c) ** (2.0 / 3.0) * np.sin(phi) ** (-1.0 / 3.0)

    assert np.allclose(Y[:, 2], expected_eta_v, rtol=1e-12, atol=1e-12)


def test_aerodynamic_outputs_match_modified_newtonian_plus_friction_formula():
    case = Waverider4D()
    X_phys = np.array([[8.0, 12.0, 2.0, 60.0]], dtype=float)
    y = case.evaluate(case.normalize(X_phys))[0]

    mach = X_phys[:, 0]
    theta_c = np.deg2rad(X_phys[:, 1])
    alpha = np.deg2rad(X_phys[:, 2])
    phi = np.deg2rad(X_phys[:, 3])
    delta = theta_c + alpha
    cp_lower = Waverider4D._cp_max(mach) * np.sin(delta) ** 2
    f_plan = np.sin(phi)
    cl_expected = (cp_lower * np.cos(delta) * f_plan)[0]
    cd_wave = (cp_lower * np.sin(delta) * f_plan)[0]
    cf = (0.0015 * (1.0 + 0.02 * mach**2) ** (-0.3))[0]
    cd_fric = cf * (2.0 / np.sin(phi))[0]
    ld_expected = cl_expected / (cd_wave + cd_fric)

    assert np.isclose(y[1], cl_expected, rtol=1e-12, atol=1e-12)
    assert np.isclose(y[0], ld_expected, rtol=1e-12, atol=1e-12)
    assert y[0] < 1.0 / np.tan(delta[0])


def test_legal_domain_does_not_need_wetted_area_or_ld_denominator_clamps():
    case = Waverider4D()
    bounds = [(var.lower, var.upper) for var in case.meta.input_variables]
    rng = np.random.default_rng(20260606)
    X_phys = np.vstack([
        np.array(list(itertools.product(*bounds)), dtype=float),
        np.column_stack([rng.uniform(lo, hi, 256) for lo, hi in bounds]),
    ])

    mach = X_phys[:, 0]
    theta_c = np.deg2rad(X_phys[:, 1])
    alpha = np.deg2rad(X_phys[:, 2])
    phi = np.deg2rad(X_phys[:, 3])
    delta = theta_c + alpha
    cp_lower = Waverider4D._cp_max(mach) * np.sin(delta) ** 2
    cd_wave = cp_lower * np.sin(delta) * np.sin(phi)
    cf = 0.0015 * (1.0 + 0.02 * mach**2) ** (-0.3)
    cd_fric = cf * 2.0 / np.sin(phi)
    cd_total = cd_wave + cd_fric

    assert np.min(np.sin(phi)) >= 0.5 - 1e-12
    assert np.all(cd_total > 1e-4)


def test_waverider_rejects_nonphysical_drag_instead_of_clipping(monkeypatch):
    case = Waverider4D()

    monkeypatch.setattr(Waverider4D, "_cp_max", staticmethod(lambda mach: -100.0 * np.ones_like(mach)))

    with pytest.raises(RuntimeError, match="non-positive total drag"):
        case._waverider_perf(
            np.array([8.0]),
            np.deg2rad(np.array([10.0])),
            np.deg2rad(np.array([0.0])),
            np.deg2rad(np.array([60.0])),
        )


def test_waverider_rejects_out_of_contract_inputs_before_formula_evaluation():
    case = Waverider4D()

    with pytest.raises(RuntimeError, match="supersonic Rayleigh-Pitot contract"):
        case._waverider_perf(
            np.array([1.0]),
            np.deg2rad(np.array([10.0])),
            np.deg2rad(np.array([0.0])),
            np.deg2rad(np.array([60.0])),
        )

    with pytest.raises(RuntimeError, match="supersonic Rayleigh-Pitot contract"):
        case._waverider_perf(
            np.array([8.0]),
            np.array([np.nan]),
            np.deg2rad(np.array([0.0])),
            np.deg2rad(np.array([60.0])),
        )


def test_declared_output_ranges_cover_waverider_domain_corners():
    case = Waverider4D()
    bounds = [(var.lower, var.upper) for var in case.meta.input_variables]
    X_phys = np.array(list(itertools.product(*bounds)), dtype=float)
    Y = case.evaluate(case.normalize(X_phys))

    for index, output in enumerate(case.meta.output_variables):
        lower, upper = output.expected_range
        values = Y[:, index]
        assert np.nanmin(values) >= lower
        assert np.nanmax(values) <= upper


def test_validation_material_declares_uncalibrated_surrogate_scope():
    validation_text = Path("tests/_cross_validation/waverider_4d.py").read_text(
        encoding="utf-8"
    )
    source_text = Path("doe_testbed/cases/aerodynamics/waverider_4d.py").read_text(
        encoding="utf-8"
    )

    assert "modified-Newtonian surrogate" in validation_text
    assert "quantitative external validation" in validation_text
    assert "10%" not in validation_text
    assert "Target accuracy" not in validation_text
    assert "not a full van Driest II" in source_text
    assert "wind-tunnel-calibrated" in validation_text
    assert "cannot replace real CFD / wind-tunnel data" in source_text
    assert "is the ideal aerodynamic shape" not in source_text
    assert "Real integrated vehicles still depend" in source_text
    normalized_source = " ".join(source_text.split())
    for omitted_scope in [
        "osculating-cone",
        "streamline-tracing",
        "shock fitting",
        "shock-attachment verification",
        "leading-edge bluntness",
        "boundary-layer heating",
        "base drag",
        "propulsion integration",
        "trim/control moments",
        "static/dynamic stability",
        "rarefied-flow effects",
        "real-gas chemistry",
        "vibrational excitation",
        "dissociation",
        "ionization",
        "chemical/thermal nonequilibrium",
        "plasma",
        "thermal protection",
    ]:
        assert omitted_scope in normalized_source
    normalized_validation = " ".join(validation_text.split())
    for source_anchor in [
        "https://ntrs.nasa.gov/api/citations/19960045290/downloads/19960045290.pdf",
        "https://ntrs.nasa.gov/citations/19960045290",
        "https://ntrs.nasa.gov/api/citations/19910013767/downloads/19910013767.pdf",
        "https://www.nasa.gov/wp-content/uploads/2023/03/equations-tables-charts-compressibleflow-report-1135.pdf",
        "https://www.grc.nasa.gov/WWW/BGH/normal.html",
        "https://www.grc.nasa.gov/www/BGH/realgas.html",
        "https://www.grc.nasa.gov/www/BGH/mach.html",
        "https://www.grc.nasa.gov/WWW/BGH/hmission.html",
        "https://www.grc.nasa.gov/www/BGH/hihyper.html",
        "https://www.nasa.gov/reference/x-43a/",
        "https://www.nasa.gov/image-article/cfd-image-of-x-43a-mach-7-test-condition/",
        "https://www.nasa.gov/centers-and-facilities/glenn/hypersonic-tunnel-facility/",
        "https://www.nasa.gov/centers-and-facilities/glenn/1x1-supersonic-wind-tunnel/",
        "https://www.nasa.gov/ames/arc-jet-complex/interaction-heating-facility/",
        "https://software.nasa.gov/software/ARC-14294-1",
        "https://www.nasa.gov/directorates/armd/aavp/ht/",
        "https://www.nasa.gov/directorates/armd/aavp/ht/hypersonic-research-topics/",
    ]:
        assert source_anchor in source_text
    for source_anchor in [
        "https://ntrs.nasa.gov/api/citations/19960045290/downloads/19960045290.pdf",
        "https://ntrs.nasa.gov/api/citations/19910013767/downloads/19910013767.pdf",
        "https://www.nasa.gov/wp-content/uploads/2023/03/equations-tables-charts-compressibleflow-report-1135.pdf",
        "https://www.grc.nasa.gov/WWW/BGH/normal.html",
        "https://www.grc.nasa.gov/www/BGH/realgas.html",
        "https://www.grc.nasa.gov/www/BGH/mach.html",
        "https://www.grc.nasa.gov/WWW/BGH/hmission.html",
        "https://www.grc.nasa.gov/www/BGH/hihyper.html",
        "https://www.nasa.gov/reference/x-43a/",
        "https://www.nasa.gov/image-article/cfd-image-of-x-43a-mach-7-test-condition/",
        "https://www.nasa.gov/centers-and-facilities/glenn/hypersonic-tunnel-facility/",
        "https://www.nasa.gov/centers-and-facilities/glenn/1x1-supersonic-wind-tunnel/",
        "https://www.nasa.gov/ames/interaction-heating-facility/",
        "https://software.nasa.gov/software/ARC-15779-1A",
        "https://www.nasa.gov/directorates/armd/aavp/ht/",
        "https://www.nasa.gov/directorates/armd/aavp/ht/hypersonic-research-topics/",
    ]:
        assert source_anchor in validation_text
    assert "public NASA anchors support waverider order-of-magnitude context" in normalized_source
    assert "Mach 5-7 design work must evaluate real-gas effects" in normalized_source
    assert "Ma > 5 hypersonic classification" in normalized_source
    assert "shock/high-temperature gas-effect boundary" in normalized_validation
    assert "design Mach number of 4.0" in normalized_source
    assert "Mach 1.6-4.63" in normalized_source
    assert "below this benchmark case's Mach 5-12 input box" in normalized_source
    assert "full Mach-domain response" in normalized_source
    assert "These anchors are not external validation data for this fixed-gamma" in normalized_source
    assert "refusing silent wetted-area clipping" in normalized_source
    assert "refusing silent L/D denominator clipping" in normalized_source
    assert "The public NASA/NACA anchors are order-of-magnitude" in normalized_validation
    assert "NASA Glenn Mach-number source is likewise classification and scope evidence only" in normalized_validation
    assert "does not validate the Rayleigh-Pitot Cpmax implementation" in normalized_validation
    assert "design Mach number of 4.0" in normalized_validation
    assert "Mach 1.6-4.63" in normalized_validation
    assert "below this case's registered Mach 5-12 box" in normalized_validation
    assert "rather than domain-matched validation" in normalized_validation
    assert "compressible-flow formula" in normalized_validation
    assert "Rayleigh-Pitot context" in normalized_validation
    assert "vibrational excitation" in normalized_validation
    assert "dissociation" in normalized_validation
    assert "ionization" in normalized_validation
    assert "chemical/thermal nonequilibrium" in normalized_validation
    assert "Mach 5-7 vehicle design must evaluate real-gas effects" in normalized_validation
    assert "strong shock waves" in normalized_validation
    assert "plasma" in normalized_validation
    assert "thermal protection" in normalized_validation
    assert "airframe-integrated scramjet" in normalized_source
    assert "high-enthalpy facility simulation" in normalized_source
    assert "Mach 5-7 true-enthalpy ground-test capability" in normalized_source
    assert "benchmark-quality experiments for CFD code validation" in normalized_source
    assert "continuous Mach 1.3-6.0 operation" in normalized_source
    assert "convective/radiative heating" in normalized_source
    assert "material thermal response" in normalized_source
    assert "ablation recession" in normalized_source
    assert "engine-operating CFD" in normalized_validation
    assert "Mach 5-7 true-enthalpy ground testing" in normalized_validation
    assert "benchmark-quality CFD-validation experiments" in normalized_validation
    assert "continuous Mach 1.3-6.0 operation" in normalized_validation
    assert "aerothermal qualification" in normalized_validation
    assert "thermal-protection testing" in normalized_validation
    assert "surface heat flux" in normalized_validation
    assert "thermal-response/ablation analysis" in normalized_validation
    assert "not validation data for the registered response surface" in normalized_validation
    assert "not a Hyper-X/X-43A airframe-integrated scramjet" in normalized_validation
    assert "not a NASA Glenn 1x1-style benchmark-quality CFD-validation experiment" in normalized_validation
    assert "not a NASA Ames arc-jet TPS test" in normalized_validation
    assert "not a FIAT-style thermal-response/ablation analysis" in normalized_validation
    assert "Hypersonic Technology Project" in normalized_source
    assert "system-level design, analysis, and validation" in normalized_validation
    assert "propulsion, vehicle-technology, or high-temperature durable-material workflows" in normalized_validation
    assert "not external validation data for this fixed-gamma modified-Newtonian surrogate" in normalized_validation
    assert "full Mach-domain response" in normalized_validation
    assert "rejects malformed arrays" in normalized_validation
    assert "before modified-Newtonian formula evaluation" in normalized_validation
    assert "cannot extrapolate beyond the registered hypersonic surrogate bounds" in normalized_validation


def test_readme_labels_waverider_as_hypersonic_not_supersonic():
    readme_en = Path("README.md").read_text(encoding="utf-8")
    readme_zh = Path("README_zh.md").read_text(encoding="utf-8")

    en_row = next(line for line in readme_en.splitlines() if line.startswith("| `waverider_4d` |"))
    zh_row = next(line for line in readme_zh.splitlines() if line.startswith("| `waverider_4d` |"))

    assert "hypersonic" in en_row.lower()
    assert "supersonic" not in en_row.lower()
    assert "高超声速" in zh_row
    assert "乘波体超声速" not in zh_row


def test_reviewer_markdown_note_keeps_waverider_p2_scope_boundary():
    note = Path("tests/_cross_validation/waverider_4d.md")
    text = note.read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "P2 plausibility-only" in normalized
    assert "No response-surface change" in normalized
    assert "Fixed-gamma modified-Newtonian impact surrogate" in normalized
    assert "Rayleigh-Pitot `Cpmax`" in normalized
    assert "fixed `gamma = 1.4` calorically perfect gas model" in normalized
    assert "approaches about `1.839`" in normalized
    assert "not CFD-validated" in normalized
    assert "not wind-tunnel-calibrated" in normalized
    assert "not Hyper-X/X-43A-equivalent" in normalized
    assert "not full Mach 5-12 validated" in normalized
    assert "not shock-attachment verified" in normalized
    assert "not a real-gas/aerothermal solver" in normalized
    assert "vibrational excitation" in normalized
    assert "dissociation" in normalized
    assert "ionization" in normalized
    assert "chemical/thermal nonequilibrium" in normalized
    assert "Mach 5-7 true-enthalpy ground testing" in normalized
    for source_anchor in [
        "https://ntrs.nasa.gov/api/citations/19960045290/downloads/19960045290.pdf",
        "https://ntrs.nasa.gov/api/citations/19910013767/downloads/19910013767.pdf",
        "https://www.nasa.gov/wp-content/uploads/2023/03/equations-tables-charts-compressibleflow-report-1135.pdf",
        "https://www.grc.nasa.gov/WWW/BGH/normal.html",
        "https://www.grc.nasa.gov/www/BGH/realgas.html",
        "https://www.grc.nasa.gov/www/BGH/mach.html",
        "https://www.grc.nasa.gov/WWW/BGH/hmission.html",
        "https://www.grc.nasa.gov/www/BGH/hihyper.html",
        "https://www.nasa.gov/reference/x-43a/",
        "https://www.nasa.gov/image-article/cfd-image-of-x-43a-mach-7-test-condition/",
        "https://www.nasa.gov/centers-and-facilities/glenn/hypersonic-tunnel-facility/",
        "https://www.nasa.gov/centers-and-facilities/glenn/1x1-supersonic-wind-tunnel/",
        "https://www.nasa.gov/ames/interaction-heating-facility/",
        "https://software.nasa.gov/software/ARC-15779-1A",
        "https://www.nasa.gov/directorates/armd/aavp/ht/",
        "https://www.nasa.gov/directorates/armd/aavp/ht/hypersonic-research-topics/",
    ]:
        assert source_anchor in text
    assert "benchmark-quality CFD-validation experiments" in normalized
    assert "continuous Mach 1.3-6.0 operation" in normalized
    assert "NASA Glenn 1x1-style wind-tunnel database comparison" in normalized
    assert "NASA Ames arc-jet TPS testing" in normalized
    assert "FIAT-style thermal-response/ablation analysis" in normalized
    assert "NASA Hypersonic Technology Project-style system-level" in normalized
    assert "high-temperature durable-material workflows" in normalized


def test_web_audit_keeps_waverider_evidence_tier_and_scope_honest():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    normalized_audit = " ".join(audit_text.split())
    row = next(
        line
        for line in audit_text.splitlines()
        if line.startswith("| `waverider_4d` |")
        and "NASA TP-3559 supports the order-of-magnitude L/D context" in line
    )

    assert "Evidence tier remains P2" in row
    assert "NASA TP-3559 supports the order-of-magnitude L/D context" in row
    assert "design Mach 4.0" in row
    assert "Mach 1.6-4.63" in row
    assert "below this benchmark case's Mach 5-12 domain" in row
    assert "not domain-matched validation" in row
    assert "integrated-vehicle details, base drag, shock attachment" in row
    assert "NASA CR-182076 supports modified-Newtonian/impact-type methods" in row
    assert "not calibrated waverider performance data" in row
    assert "NASA Glenn real-gas material is boundary evidence" in row
    assert "NACA Report 1135" in row
    assert "NASA Glenn normal-shock material" in row
    assert "Rayleigh-Pitot/Cpmax formula context" in row
    assert "fixed-`gamma=1.4` calorically-perfect surrogate" in row
    assert "not CFD, wind-tunnel, full Mach-domain, shock-attached-design" in row
    assert "real-gas/aerothermal, trim/stability, or integrated-vehicle validation" in row
    assert "Keep P2 plausibility-only" in row
    assert "full Mach 5-12 validation" in row
    assert "no additional numerical resweep" in row

    assert "Additional `waverider_4d` real-gas boundary note" in audit_text
    assert "vibrational excitation" in audit_text
    assert "dissociation" in audit_text
    assert "ionization" in audit_text
    assert "chemical/thermal nonequilibrium" in audit_text
    assert "Additional `waverider_4d` 2026-06-08 official-source re-check" in audit_text
    assert "formula, order-of-magnitude, and omitted-scope anchors" in normalized_audit
    assert "not as external validation data for the registered response surface" in normalized_audit
    assert "base drag, viscous effects, shock attachment, and component build-up" in normalized_audit
    assert "design Mach 4.0 and data over Mach 1.6-4.63" in normalized_audit
    assert "below the benchmark's Mach 5-12 box" in normalized_audit
    assert "Mach 5-7 design work must evaluate real-gas effects" in normalized_audit
    assert "preliminary-analysis-system / modified-Newtonian and impact-method context" in normalized_audit
    assert "Rayleigh-Pitot / compressible flow anchors" in normalized_audit
    assert "Evidence tier remains P2 plausibility-only" in normalized_audit
    assert "Do not describe `waverider_4d` as CFD-validated" in normalized_audit
    assert "full Mach 5-12 validated" in normalized_audit
    assert "full van Driest II boundary-layer/friction validated" in normalized_audit
    assert "No numerical response resweep is needed solely for this official-source re-check" in normalized_audit
    assert (
        "Additional `waverider_4d` 2026-06-08 NASA hypersonic-flight boundary tightening:"
        in audit_text
    )
    assert "Additional `waverider_4d` 2026-06-08 hypersonic-mission boundary note:" in audit_text
    assert "mission-grade hypersonic design analysis" in normalized_audit
    assert "Navier-Stokes plus real-gas modelling" in normalized_audit
    assert "Additional `waverider_4d` 2026-06-09 NASA Glenn Mach-number boundary note" in normalized_audit
    assert "Ma > 5 hypersonic classification" in normalized_audit
    assert "shock-wave and high-temperature gas effects become central" in normalized_audit
    assert "does not validate the fixed-gamma Rayleigh-Pitot `Cpmax` implementation" in normalized_audit
    assert "high-temperature gas chemistry" in normalized_audit
    assert "strong shock waves" in normalized_audit
    assert "plasma" in normalized_audit
    assert "special thermal protection" in normalized_audit
    assert "blunt-body/re-entry drag management" in normalized_audit
    assert (
        "Additional `waverider_4d` 2026-06-09 NASA Glenn 1x1 CFD-validation "
        "facility boundary note"
        in normalized_audit
    )
    assert "continuous Mach 1.3-6.0 operation" in normalized_audit
    assert "benchmark-quality experiments for CFD code validation" in normalized_audit
    assert "does not compare against a NASA Glenn 1x1 database" in normalized_audit
    assert "run CFD-code validation" in normalized_audit
    assert "facility Reynolds-number/enthalpy matching" in normalized_audit
    assert "wind-tunnel-calibrated performance" in normalized_audit
    assert (
        "Additional `waverider_4d` 2026-06-09 NASA Ames arc-jet/TPS boundary note"
        in normalized_audit
    )
    assert "Interaction Heating Facility / Arc Jet Complex" in normalized_audit
    assert "FIAT thermal-response software" in normalized_audit
    assert "convective/radiative heating environments" in normalized_audit
    assert "surface heat-flux and thermal-protection-system testing" in normalized_audit
    assert "material thermal-response calculation" in normalized_audit
    assert "ablation/recession analysis" in normalized_audit
    assert "does not run arc-jet TPS testing" in normalized_audit
    assert "does not compute material thermal response or ablation" in normalized_audit
    assert "Additional `waverider_4d` 2026-06-09 NASA Hypersonic Technology Project program-boundary note" in normalized_audit
    assert "reusable airbreathing hypersonic flight" in normalized_audit
    assert "system-level design, analysis and validation" in normalized_audit
    assert "propulsion, vehicle technologies, and high-temperature durable materials" in normalized_audit
    assert "does not perform system-level hypersonic vehicle design or validation" in normalized_audit
    assert "HTP-style reusable airbreathing flight validation" in normalized_audit


def test_web_audit_records_waverider_current_page_spot_check():
    audit_text = Path("docs/CASE_WEB_VALIDATION_AUDIT.md").read_text(encoding="utf-8")
    normalized = " ".join(audit_text.split())

    assert (
        "Additional `waverider_4d` 2026-06-09 current-page spot check:"
        in audit_text
    )
    for phrase in (
        "current NASA TP-3559/NTRS metadata",
        "NASA CR-182076 APAS material",
        "NASA Glenn Mach-number, normal-shock, real-gas, hypersonic-mission, and re-entry pages",
        "NASA X-43A Hyper-X pages",
        "NASA Glenn Hypersonic Tunnel Facility and 1x1 Supersonic Wind Tunnel pages",
        "NASA Ames arc-jet/FIAT material",
        "NASA Hypersonic Technology Project pages",
        "order-of-magnitude waverider context",
        "modified-Newtonian / impact-method formula context",
        "Rayleigh-Pitot / normal-shock compressible-flow context",
        "Ma > 5 hypersonic label",
        "real-gas evaluation",
        "variable specific heats",
        "vibrational excitation",
        "dissociation",
        "ionization",
        "chemical/thermal nonequilibrium",
        "strong shocks",
        "plasma",
        "high-temperature durable materials",
        "airframe-integrated scramjet propulsion",
        "high-enthalpy ground testing",
        "benchmark-quality CFD-validation experiments",
        "wind-tunnel database comparison",
        "convective/radiative heating",
        "arc-jet TPS testing",
        "thermal-response modelling",
        "ablation/recession analysis",
        "system-level design/analysis/validation",
        "design Mach 4.0 and data over Mach 1.6-4.63",
        "registered over Mach 5-12",
        "four-input fixed-gamma Rayleigh-Pitot / modified-Newtonian plus empirical-friction L_D/CL/eta_v surrogate",
        "Do not describe this case as CFD validated",
        "wind-tunnel validated",
        "full Mach 5-12 validated",
        "shock-attachment verified",
        "Hyper-X/X-43A validated",
        "NASA Glenn 1x1 validated",
        "true-enthalpy facility validated",
        "arc-jet/TPS validated",
        "FIAT validated",
        "HTP validated",
        "system-level hypersonic-vehicle validated",
        "propulsion validated",
        "high-temperature-material validated",
        "real-gas validated",
        "aerothermal validated",
        "integrated-vehicle validated",
        "Evidence tier remains P2 plausibility-only",
        "No numerical response resweep is needed solely for this current-page spot check",
    ):
        assert phrase in normalized


def test_metadata_declares_full_modified_newtonian_interaction_structure():
    case = Waverider4D()
    references = "\n".join(case.meta.physics_references)
    details = case.meta.solver_details.lower()

    roles = {var.name: var.role for var in case.meta.input_variables}
    assert roles["alpha"] == "noise"
    assert case.meta.noise_factor_candidates == ["alpha"]
    assert set(case.meta.control_factor_candidates) == {"Ma", "theta_c", "phi"}
    assert "smooth_unimodal" in case.meta.response_topology
    assert "strong_interaction" in case.meta.response_topology
    assert set(case.meta.important_variables) == {"Ma", "theta_c", "alpha", "phi"}
    assert set(case.meta.interaction_pairs) == {
        ("Ma", "theta_c"),
        ("Ma", "alpha"),
        ("Ma", "phi"),
        ("theta_c", "alpha"),
            ("theta_c", "phi"),
            ("alpha", "phi"),
    }
    assert "NASA TP-3559" in references
    assert "19960045290.pdf" in references
    assert "design Mach 4.0" in references
    assert "Mach 1.6-4.63" in references
    assert "below this case's Mach 5-12 domain" in references
    assert "NASA CR-182076" in references
    assert "19910013767.pdf" in references
    assert "NACA Report 1135" in references
    assert "compressible-flow context" in references
    assert "equations-tables-charts-compressibleflow-report-1135.pdf" in references
    assert "Normal Shock Wave Equations" in references
    assert "calorically perfect/imperfect gas shock context" in references
    assert "https://www.grc.nasa.gov/WWW/BGH/normal.html" in references
    assert "https://www.grc.nasa.gov/www/BGH/index.html" in references
    assert "Real Gas Effects" in references
    assert "https://www.grc.nasa.gov/www/BGH/realgas.html" in references
    assert "Hypersonic Missions" in references
    assert "Mach Number" in references
    assert "https://www.grc.nasa.gov/www/BGH/mach.html" in references
    assert "Ma > 5 hypersonic classification" in references
    assert "https://www.grc.nasa.gov/WWW/BGH/hmission.html" in references
    assert "Re-Entry Aircraft" in references
    assert "https://www.grc.nasa.gov/www/BGH/hihyper.html" in references
    assert "1x1 Supersonic Wind Tunnel" in references
    assert "continuous Mach 1.3-6.0 operation" in references
    assert "https://www.nasa.gov/centers-and-facilities/glenn/1x1-supersonic-wind-tunnel/" in references
    assert "Interaction Heating Facility" in references
    assert "https://www.nasa.gov/ames/arc-jet-complex/interaction-heating-facility/" in references
    assert "convective/radiative heating" in references
    assert "Fully Implicit Ablation and Thermal Response Program" in references
    assert "https://software.nasa.gov/software/ARC-14294-1" in references
    assert "thermal-response and ablation-analysis software boundary context" in references
    assert "NASA Hypersonic Technology Project" in references
    assert "https://www.nasa.gov/directorates/armd/aavp/ht/" in references
    assert "https://www.nasa.gov/directorates/armd/aavp/ht/hypersonic-research-topics/" in references
    assert "system-level design/analysis/validation" in references
    assert "high-temperature durable materials" in references
    for omitted_scope in [
        "osculating-cone",
        "streamline-tracing",
        "shock fitting",
        "shock-attachment verification",
        "leading-edge bluntness",
        "boundary-layer heating",
        "base drag",
        "propulsion integration",
        "trim/control moments",
        "real-gas chemistry",
        "nasa glenn mach-number shock/high-temperature gas-effect workflow calibration",
        "rarefied-flow effects",
        "static/dynamic stability",
        "tps",
        "benchmark-quality cfd-validation experiments",
        "arc-jet thermal-protection testing",
        "material thermal-response",
        "ablation analysis",
        "nasa hypersonic technology project system-level design/analysis/validation",
        "propulsion",
        "vehicle-technologies",
        "high-temperature durable-materials workflow",
        "cfd",
        "wind-tunnel calibration",
    ]:
        assert omitted_scope in details
    assert "fixed gamma=1.4" in details
    assert "calorically-perfect-gas cpmax" in details
    assert "variable specific heats" in details
    assert "vibrational excitation" in details
    assert "dissociation" in details
    assert "ionization" in details
    assert "chemical/thermal nonequilibrium" in details
    assert "nonequilibrium chemistry" in details
    assert "plasma effects" in details
    assert "blunt-body/re-entry high-drag flight" in details
    assert hasattr(case, "_validate_physical_inputs")
    assert "named _validate_physical_inputs" in details
    assert "rejects malformed arrays" in details
    assert "before modified-newtonian formula evaluation" in details
    assert "out-of-contract l_d/cl/eta_v responses are rejected" in details


def test_scenario_notes_do_not_overclaim_mach_interaction_strength():
    case = Waverider4D()
    note = case.meta.scenario_notes["VS-B"]

    assert "Geometry dominates" in note
    assert "Ma is a weak hypersonic trend variable" in note
    assert "dominant interaction driver" in note
    assert case.meta.sobol_ST["Ma"] < 0.02


def test_applicable_scenarios_have_non_overclaiming_notes():
    case = Waverider4D()

    assert set(case.meta.applicable_scenarios) <= set(case.meta.scenario_notes)
    vs_c = case.meta.scenario_notes["VS-C"]
    vs_d = case.meta.scenario_notes["VS-D"]
    assert "not CFD- or wind-tunnel-calibrated prediction accuracy" in vs_c
    assert "calorically-perfect" in vs_c
    assert "variable-specific-heat" in vs_c
    assert "vibrational excitation" in vs_c
    assert "dissociation" in vs_c
    assert "ionization" in vs_c
    assert "chemical/thermal nonequilibrium chemistry" in vs_c
    assert "shock-attached design" in vs_c
    assert "shock-attachment verification" in vs_c
    assert "real-gas/aerothermal analysis" in vs_c
    assert "rarefied-flow analysis" in vs_c
    assert "trim/control stability" in vs_c
    assert "full integrated-vehicle performance" in vs_c
    assert "small-sample" in vs_d
    assert "without pre-claiming a required sample count" in vs_d


def test_modified_newtonian_solver_failure_is_not_silent_nan(monkeypatch):
    case = Waverider4D()

    def invalid_response(*_args, **_kwargs):
        return np.array([[np.nan, 0.1, 0.05]])

    monkeypatch.setattr(case, "_waverider_perf", invalid_response)
    X_phys = np.array([[8.0, 12.0, 2.0, 60.0]], dtype=float)

    with pytest.raises(RuntimeError, match="refusing silent non-finite"):
        case._evaluate_core(X_phys)


def test_nonpositive_modified_newtonian_response_is_not_silent_archive_row(monkeypatch):
    case = Waverider4D()

    def invalid_response(*_args, **_kwargs):
        return np.array([[2.0, 0.0, 0.05]])

    monkeypatch.setattr(case, "_waverider_perf", invalid_response)
    X_phys = np.array([[8.0, 12.0, 2.0, 60.0]], dtype=float)

    with pytest.raises(RuntimeError, match="non-positive"):
        case._evaluate_core(X_phys)


def test_out_of_contract_modified_newtonian_response_is_not_archived(monkeypatch):
    case = Waverider4D()
    X_phys = np.array([[8.0, 12.0, 2.0, 60.0]], dtype=float)

    invalid_responses = [
        np.array([[0.49, 0.1, 0.05]]),
        np.array([[10.01, 0.1, 0.05]]),
        np.array([[2.0, 0.51, 0.05]]),
        np.array([[2.0, 0.1, 0.151]]),
    ]

    for response in invalid_responses:
        monkeypatch.setattr(case, "_waverider_perf", lambda *_args, **_kwargs: response)
        with pytest.raises(RuntimeError, match="declared output ranges"):
            case._evaluate_core(X_phys)


def test_named_private_validator_rejects_malformed_or_out_of_domain_inputs_before_formula(monkeypatch):
    case = Waverider4D()
    assert hasattr(case, "_validate_physical_inputs")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("waverider formula should not be called for invalid physical input")

    monkeypatch.setattr(case, "_waverider_perf", fail_if_called)

    with pytest.raises(ValueError, match="shape"):
        case._evaluate_core(np.array([8.0, 12.0, 2.0, 60.0], dtype=float))

    invalid_rows = [
        [4.99, 12.0, 2.0, 60.0],
        [12.01, 12.0, 2.0, 60.0],
        [8.0, 4.99, 2.0, 60.0],
        [8.0, 20.01, 2.0, 60.0],
        [8.0, 12.0, -0.1, 60.0],
        [8.0, 12.0, 5.1, 60.0],
        [8.0, 12.0, 2.0, 29.9],
        [8.0, 12.0, 2.0, 90.1],
        [8.0, np.nan, 2.0, 60.0],
    ]
    for row in invalid_rows:
        with pytest.raises(ValueError, match="Ma/theta_c/alpha/phi benchmark domain"):
            case._evaluate_core(np.array([row], dtype=float))
