# Evidence note: plate_with_hole_5d

Audit date: 2026-09-17. This note documents the current implementation; it does not modify or certify the archived response data.

- case_id: `plate_with_hole_5d`
- source_module: `doe_testbed/cases/structures/plate_with_hole_5d.py`
- targeted_testfile: `tests/test_plate_with_hole_physics.py`
- cross_validation_testfile: `tests/_cross_validation/plate_with_hole_5d.py`
- status_source: `docs/case_validation_status.csv`; risk `P1`; disposition: **accepted after resweep**.

## Implemented model and parameters

Inputs: axis ratio q=a/b=0.5–5, size parameter a_over_W=0.1–0.45, σx=50–300 MPa, σy=−50–100 MPa and angle=0–90°. The code rotates the stress tensor, uses Kx=1+2 q, Ky=1+2/q, Ks=2(1+q), and multiplies by H=1+0.5 r³+0.3 r⁵, r=min(2*a_over_W,0.75). It returns σ=max(|σx_eff Kx H−σy_eff|,|σy_eff Ky H−σx_eff|)+|τxy_eff Ks H| (floored at 0.1 MPa), Kt=σ/max(|σx_eff|,|σy_eff|,1), τmax=σ/2.

## Traceable external evidence

[Wiley's Peterson frontmatter record](https://onlinelibrary.wiley.com/doi/10.1002/9780470211106.fmatter) confirms Pilkey/Pilkey third edition, ©2008, ISBN 9780470048245; reused from the 2026-09-16 audit. Kirsch's article identity is corroborated by [CiNii](https://cir.nii.ac.jp/crid/1573105975750615168), but the original article was not inspected. **No original page/chart was verified for the stated Heywood polynomial, shear multiplier, or additive maximum rule.** Inglis bibliographic page range and the claimed Pilkey Chart 4.1/Table 4-1 attribution remain unresolved.

Supplementary internal audit history (not required to interpret this note; may be absent from the public source package): `paper-reviews/communications-engineering/2026-09-16/case_sources_01_15.md` and `case_sources_31_43.md`. Bibliographic existence and equation-level verification are distinct. No unstated book pages, figure numbers or measured reference data are claimed here.

## Local tests and what they establish

`test_oblique_loading_includes_inglis_shear_response` reproduces the implemented expression at q=3, a_over_W=0.2, σx=200, σy=0, θ=45°, rtol 1e-12. `test_finite_width_correction_caps_heywood_argument` pins the 0.75 cap. Cross-validation `test_kirsch_circular_hole` checks an approximately threefold uniaxial concentration at small finite-width correction. This is self-consistency of a chosen response formula, not independent finite-plate elasticity validation.

Test execution on 2026-09-17: the targeted test file named above passed all 5 collected tests in a combined 10-file run (163 tests total, exit code 0). Cross-validation functions were inspected but not rerun in this pass. The public `docs/CASE_EVIDENCE_INDEX.md` also summarizes test execution; the present note records its own scope and result. Metadata/string tests check documentation contracts, not physical accuracy. Input validation and finite-output tests protect the computational contract, not real-world predictive validity.

## Limits and documentation discrepancies

Calling the complete output an exact hole-edge maximum is not established: adding separate normal/shear maxima does not solve the angular stress maximum. The docstring mixes full width and half-width definitions; the operative rule is explicitly 2*a_over_W. Also q<1 is allowed despite calling a the semi-major axis. Circular-hole rotational invariance is not established by the current tests. Keep the polynomial and oblique combination labelled as unverified surrogate choices until original formula pages or an independent elasticity/FE comparison are supplied.

**中文证据边界：** 本页区分代码实际方程、可追溯出版来源和本地测试范围。公式回归、解析极限或数值自洽不等于真实试验校准；未读取的原书页码和未验证的经验系数均不视为已证实。既有 P1/P2 状态未升级，代码及归档未改动。
