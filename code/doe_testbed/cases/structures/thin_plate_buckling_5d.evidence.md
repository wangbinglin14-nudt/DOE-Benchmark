# Evidence note: thin_plate_buckling_5d

Audit date: 2026-09-17. This note documents the current implementation; it does not modify or certify the archived response data.

- case_id: `thin_plate_buckling_5d`
- source_module: `doe_testbed/cases/structures/thin_plate_buckling_5d.py`
- targeted_testfile: `tests/test_thin_plate_buckling_physics.py`
- cross_validation_testfile: `tests/_cross_validation/thin_plate_buckling_5d.py`
- status_source: `docs/case_validation_status.csv`; risk `P1`; disposition: **accepted after resweep**.

## Implemented model and parameters

With unit width b=1 m, λ=a/b=0.5–4, h/b=0.005–0.05, E=50–300 GPa, ν=0.25–0.45 and c=bc_flag∈[0,1], D=Eh³/[12(1−ν²)]. SSSS uses k 0=min(m/λ+λ/m)² over integers m=1…20. The other branch sets k 1=6.97 for λ≥1.4, otherwise clip(21.03λ²−46.96λ+32.22,6.97,15). k=(1−c)k 0+ck 1; Ncr=kπ²D/b²; σcr=Ncr/h in MPa.

## Traceable external evidence

Timoshenko/Gere 2 nd edition (1961) and Brush/Almroth (1975) were bibliographically verified in the previous audit; [Finnish library record](https://kansalliskirjasto.finna.fi/Record/vaari.858012) establishes the former edition. [NASA's rotational-restraint plate report](https://ntrs.nasa.gov/citations/19930091813) confirms the distinct role of restraint along unloaded edges (record/abstract checked 2026-09-17). It does **not** verify this quadratic fit, breakpoint, cap or interpolation. Claimed book chapter/section numbers remain unverified.

Supplementary internal audit history (not required to interpret this note; may be absent from the public source package): `paper-reviews/communications-engineering/2026-09-16/case_sources_01_15.md` and `case_sources_31_43.md`. Bibliographic existence and equation-level verification are distinct. No unstated book pages, figure numbers or measured reference data are claimed here.

## Local tests and what they establish

`test_ssss_outputs_match_independent_kirchhoff_formula` and cross-validation `test_square_plate_ssss`, `test_rectangular_2to1_ssss`, `test_rectangular_1p5_ssss` check k=4,4, and about 4.3403 respectively. `test_scsc_short_plate_coefficient_matches_validation_note` only bounds k at λ=0.5 between 13 and 14.2; it does not compare a measured or independently digitized table. `test_continuous_bc_flag_interpolates_elastic_restraint_surrogate` verifies the selected interpolation.

Test execution on 2026-09-17: the targeted test file named above passed all 9 collected tests in a combined 10-file run (163 tests total, exit code 0). Cross-validation functions were inspected but not rerun in this pass. The public `docs/CASE_EVIDENCE_INDEX.md` also summarizes test execution; the present note records its own scope and result. Metadata/string tests check documentation contracts, not physical accuracy. Input validation and finite-output tests protect the computational contract, not real-world predictive validity.

## Limits and documentation discrepancies

Source introduction says the two short edges are clamped, whereas implementation comments say unloaded longitudinal edges are clamped. That convention must be reconciled before physical design use. Continuous c is a numerical blend, with no demonstrated mapping to actual rotational-spring stiffness. The piecewise fit may introduce a jump at λ=1.4 (left quadratic≈7.4948 versus 6.97). No imperfection sensitivity, plasticity, post-buckling or fixture calibration is established.

**中文证据边界：** 本页区分代码实际方程、可追溯出版来源和本地测试范围。公式回归、解析极限或数值自洽不等于真实试验校准；未读取的原书页码和未验证的经验系数均不视为已证实。既有 P1/P2 状态未升级，代码及归档未改动。
