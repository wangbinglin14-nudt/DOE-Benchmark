# Evidence note: groundwater_contamination_6d

Audit date: 2026-09-17. This note documents the current implementation; it does not modify or certify the archived response data.

- case_id: `groundwater_contamination_6d`
- source_module: `doe_testbed/cases/geomechanics/groundwater_contamination_6d.py`
- targeted_testfile: `tests/test_groundwater_contamination_physics.py`
- cross_validation_testfile: `tests/_cross_validation/groundwater_contamination_6d.py`
- status_source: `docs/case_validation_status.csv`; risk `P2`; disposition: **plausibility only**.

## Implemented model and parameters

v=.001–1 m/d, α=.01–5 m, Kd=0–10 L/kg, ρb=1400–1900 kg/m³, half-life 10–1000 d, porosity .2–.5. D=αv+.7(1e−9×86400)m²/d, R=1+ρbKd/(1000 n), λ=ln 2/half-life. C/C 0=.5[erfc((Rx−vt)/(2√DRt))+exp(vx/D)erfc((Rx+vt)/(2√DRt))]exp(−λt/R), using erfcx for stable reflection evaluation. At 365 d,500 spatial samples 0–500 m give source-including Cmax and the furthest sampled C>.01 point. Arrival at 10 m is first threshold crossing among 200 log-spaced times 1–10000 d;10000 also means no detected crossing.

## Traceable external evidence

[USGS original report PP 411-A](https://pubs.usgs.gov/publication/pp411A) confirms Ogata/Banks 1961, pagesA 1–A 7, DOI 10.3133/pp 411 A (official record checked 2026-09-17). The report's specific equation pages were not read in this pass. It anchors the nondecaying 1 D dispersion family, **not** the appended exp(−λt/R) as an exact constant-source reactive solution. Domenico/Schwartz §17.3/Example 17.2/p 413 were explicitly unverified in the previous bibliography audit.

Supplementary internal audit history (not required to interpret this note; may be absent from the public source package): `paper-reviews/communications-engineering/2026-09-16/case_sources_01_15.md` and `case_sources_31_43.md`. Bibliographic existence and equation-level verification are distinct. No unstated book pages, figure numbers or measured reference data are claimed here.

## Local tests and what they establish

`test_relative_concentration_matches_declared_ogata_banks_surrogate_formula` checks the selected formula at x 10 m, t 180 d, R 5 to 1e−12. `test_relative_concentration_uses_stable_reflection_term_without_exponent_cap` checks numerical stability. `test_arrival_time_detects_transient_threshold_crossing_before_decay` verifies first-crossing semantics. Cross-validation `test_no_adsorption_no_decay` only checks broad bounds: half-life 999 d is not infinite (atR 1,365 d attenuation≈.776). `test_longer_time_more_spread` actually changes velocity, not observation time.

Test execution on 2026-09-17: the targeted test file named above passed all 33 collected tests in a combined 10-file run (163 tests total, exit code 0). Cross-validation functions were inspected but not rerun in this pass. The public `docs/CASE_EVIDENCE_INDEX.md` also summarizes test execution; the present note records its own scope and result. Metadata/string tests check documentation contracts, not physical accuracy. Input validation and finite-output tests protect the computational contract, not real-world predictive validity.

## Limits and documentation discrepancies

CSV:P2 plausibility-only. At x 0 the implemented concentration equals exp(−λt/R), so it is not a maintained constant-concentration source when decay is nonzero. Cmax includes that source and can be independent of v and α; do not call it exposure at 10 m. Front/arrival are quantized and the 10000 d sentinel is ambiguous. No field plume, source history,3 D flow or full reactive-transport calibration is represented.

**中文证据边界：** 本页区分代码实际方程、可追溯出版来源和本地测试范围。公式回归、解析极限或数值自洽不等于真实试验校准；未读取的原书页码和未验证的经验系数均不视为已证实。既有 P1/P2 状态未升级，代码及归档未改动。
