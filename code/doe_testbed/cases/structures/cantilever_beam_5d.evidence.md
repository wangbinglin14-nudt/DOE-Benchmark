# Evidence note: cantilever_beam_5d

Audit date: 2026-09-17. This note documents the current implementation; it does not modify or certify the archived response data.

- case_id: `cantilever_beam_5d`
- source_module: `doe_testbed/cases/structures/cantilever_beam_5d.py`
- targeted_testfile: `tests/test_cantilever_beam_physics.py`
- cross_validation_testfile: `tests/_cross_validation/cantilever_beam_5d.py`
- status_source: `docs/case_validation_status.csv`; risk `P1`; disposition: **accepted after resweep**.

## Implemented model and parameters

The implementation evaluates σ=6 FL/(bh²), τ=3 F/(2 bh), and δ=FL³/(3 EI)+FL/(κGA), with I=bh³/12, A=bh, ν=0.30, κ=5/6 and G=E/[2(1+ν)]. It converts kN→N, GPa→Pa, stress→MPa and displacement→mm. Inputs are F=1–50 kN, L=0.5–3 m, b=0.05–0.30 m, h=0.10–0.50 m and E=60–210 GPa. Outputs are bending stress, tip deflection and neutral-axis shear stress. The 250 MPa and 20 mm thresholds are benchmark choices, not material certification.

## Traceable external evidence

[MIT Structural Mechanics recitation](https://ocw.mit.edu/courses/2-080j-structural-mechanics-fall-2013/de27f1d8f647ff995771d4b8d48d34bc_MIT2_080JF13_Recitation5.pdf) independently lists the end-load Euler–Bernoulli deflection PL³/(3 EI) (indexed content checked 2026-09-17; no page number asserted). Roark 7 th edition, 2002, and Shigley 11 th edition, 2020, were bibliographically checked in the 2026-09-16 audit; [McGraw Hill](https://www.mheducation.com/unitas/highered/changes/budynas-mech-engineering-design-11e.pdf) confirms the latter ISBN 9780073398211. Table A-9 / Roark Table 8.1 and the precise shear-correction source pages were not inspected; they are not verified equation locators.

Supplementary internal audit history (not required to interpret this note; may be absent from the public source package): `paper-reviews/communications-engineering/2026-09-16/case_sources_01_15.md` and `case_sources_31_43.md`. Bibliographic existence and equation-level verification are distinct. No unstated book pages, figure numbers or measured reference data are claimed here.

## Local tests and what they establish

`test_short_deep_beam_deflection_includes_timoshenko_shear_term` checks bending+shear at F=50 kN, L=h=0.5 m, rtol 1e-12; shear exceeds 10% of bending there. `test_shear_correction_does_not_change_bending_or_shear_stresses` isolates the modification. Cross-validation `test_known_numerical_values` checks 22.5 MPa bending stress and 0.75 MPa shear for F=10 kN, L=1.5 m, b=0.1 m, h=0.2 m, E=200 GPa. These are equation and unit controls, not measured-beam comparisons.

Test execution on 2026-09-17: the targeted test file named above passed all 7 collected tests in a combined 10-file run (163 tests total, exit code 0). Cross-validation functions were inspected but not rerun in this pass. The public `docs/CASE_EVIDENCE_INDEX.md` also summarizes test execution; the present note records its own scope and result. Metadata/string tests check documentation contracts, not physical accuracy. Input validation and finite-output tests protect the computational contract, not real-world predictive validity.

## Limits and documentation discrepancies

The docstring's “only 3 variables” sentence is wrong: stress depends on F, L, b, h, and is exactly independent only of E. L is linear in stress. The introductory torque/twist story has no torque input or twist output. The displayed combined von-Mises criterion is not evaluated; moreover bending and shear maxima occur at different section locations and cannot automatically be combined as co-located stresses. Very short/deep or large-deflection corners exceed elementary beam-theory assumptions; adding a shear term does not validate those geometries experimentally.

**中文证据边界：** 本页区分代码实际方程、可追溯出版来源和本地测试范围。公式回归、解析极限或数值自洽不等于真实试验校准；未读取的原书页码和未验证的经验系数均不视为已证实。既有 P1/P2 状态未升级，代码及归档未改动。
