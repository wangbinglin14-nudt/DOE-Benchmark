# Evidence note: dam_seepage_6d

Audit date: 2026-09-17. This note documents the current implementation; it does not modify or certify the archived response data.

- case_id: `dam_seepage_6d`
- source_module: `doe_testbed/cases/geomechanics/dam_seepage_6d.py`
- targeted_testfile: `tests/test_dam_seepage_physics.py`
- cross_validation_testfile: `tests/_cross_validation/dam_seepage_6d.py`
- status_source: `docs/case_validation_status.csv`; risk `P2`; disposition: **plausibility only**.

## Implemented model and parameters

Steady ∇·(K∇H)=0 on a 30 m-wide foundation, depth 10–50 m, fixed 20×15 grid and harmonic interface K. H 1=20–100 m, H 2=0–10 m; Kbase=1e−7…1e−4 m/s, Kcurtain=1e−9…1e−7 m/s. A full-depth curtain mask is centered 5 m from upstream, nominal width 1 m but at least one grid spacing. Drain location is grid-selected near 8 m. Relief head=H 2+(1−e)(H 1−H 2)/2 for e=0…0.8, activated only if below the no-drain local head and e>0. Side heads are prescribed; other top/bottom nodes impose zero normal flux. Outputs integrate 9.81 H over the base to MN/m, interpolate exact midpoint to MPa, and clip uplift ratio to[.05,1].

## Traceable external evidence

[FERC Gravity Dams chapter](https://www.ferc.gov/sites/default/files/2020-04/chap3.pdf), PDF pages 8–9 (zero-based 7–8), checked 2026-09-17, discusses instrumentation-supported uplift reduction and drain assumptions. This supports the **boundary on engineering claims**, not this single-node relief law. [USGS Circular 1224](https://pubs.usgs.gov/circ/2002/circ1224/html/understanding.html), indexed content checked 2026-09-17, describes Darcy proportionality. No source page has been established for this code's half-head rule or efficiency-to-head mapping. The previous audit also warns against mixing Bear 1972 with the 1988 Dover ISBN.

Supplementary internal audit history (not required to interpret this note; may be absent from the public source package): `paper-reviews/communications-engineering/2026-09-16/case_sources_01_15.md` and `case_sources_31_43.md`. Bibliographic existence and equation-level verification are distinct. No unstated book pages, figure numbers or measured reference data are claimed here.

## Local tests and what they establish

`test_homogeneous_no_drain_case_recovers_linear_uplift_control` checks Utotal=9.81(H 1+H 2)L/2000 and midpoint 9.81(H 1+H 2)/2000 at 1e−10 tolerance. `test_homogeneous_no_drain_case_has_linear_base_head_profile` checks the entire base profile. `test_drain_surrogate_never_recharges_strong_curtain_cases` checks the complementarity guard; `test_declared_curtain_width_selects_the_current_single_grid_column` pins discretized geometry. Cross-validation head/drain trend tests are plausibility checks.

Test execution on 2026-09-17: the targeted test file named above passed all 22 collected tests in a combined 10-file run (163 tests total, exit code 0). Cross-validation functions were inspected but not rerun in this pass. The public `docs/CASE_EVIDENCE_INDEX.md` also summarizes test execution; the present note records its own scope and result. Metadata/string tests check documentation contracts, not physical accuracy. Input validation and finite-output tests protect the computational contract, not real-world predictive validity.

## Limits and documentation discrepancies

The CSV retains P2 plausibility-only. `eta_uplift` is a clipped benchmark ratio, not a validated design-code coefficient despite docstring wording. Continuous e is not calibrated drain effectiveness. Fixed-grid geometry, head activation and clipping can create non-smooth behavior. No mesh-converged heterogeneous reference, measured uplift, piping assessment or dam-safety compliance has been established. SL 319 citation alone cannot certify compliance.

**中文证据边界：** 本页区分代码实际方程、可追溯出版来源和本地测试范围。公式回归、解析极限或数值自洽不等于真实试验校准；未读取的原书页码和未验证的经验系数均不视为已证实。既有 P1/P2 状态未升级，代码及归档未改动。
