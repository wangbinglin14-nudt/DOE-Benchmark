# Evidence note: soil_consolidation_5d

Audit date: 2026-09-17. This note documents the current implementation; it does not modify or certify the archived response data.

- case_id: `soil_consolidation_5d`
- source_module: `doe_testbed/cases/geomechanics/soil_consolidation_5d.py`
- targeted_testfile: `tests/test_soil_consolidation_physics.py`
- cross_validation_testfile: `tests/_cross_validation/soil_consolidation_5d.py`
- status_source: `docs/case_validation_status.csv`; risk `P2`; disposition: **analytical control; not external site-specific geotechnical validation**.

## Implemented model and parameters

Single drainage: top excess pressure 0, impermeable base atH, initially uniform surcharge Δσ. k=1e−10…1e−7 m/s, mv=1e−7…5e−6 Pa−1, H=2–20 m, Δσ=20–200 kPa, t=.5–50 yr. cv=k/(mv 9810), Tv=cv t/H² with 365.25 d/yr. For Mj=π(2 j+1)/2, j=0…1999, U=1−Σ2 exp(−Mj²Tv)/Mj². Settlement=U mv Δσ H in mm; u_mid=Σ(2Δσ/Mj)sin(Mj/2)exp(−Mj²Tv) in kPa. n_z/n_t fidelity arguments are retained but unused by the series.

## Traceable external evidence

[Itasca 1 D consolidation verification example](https://docs.itascacg.com/itasca940/flac3d/zone/test3d/Fluid/1DConsolidation/1dconsolidation.html), checked 2026-09-17, equations(2)–(4), gives diffusion, one drained/one impervious boundary and odd-mode pressure series. Its coordinate direction is reversed relative to this code and its Biot storage/initial-pressure definitions differ; it supports the mathematical family, not identical parameter calibration. Original Terzaghi/Das/Craig chapter locators were not read. The previous audit corrected Das ninth-edition authors/year and Craig ISBN; do not combine those editions' pages with the source's Das 2010 seventh-edition citation.

Supplementary internal audit history (not required to interpret this note; may be absent from the public source package): `paper-reviews/communications-engineering/2026-09-16/case_sources_01_15.md` and `case_sources_31_43.md`. Bibliographic existence and equation-level verification are distinct. No unstated book pages, figure numbers or measured reference data are claimed here.

## Local tests and what they establish

`test_degree_of_consolidation_matches_terzaghi_series` compares an independent series helper (2% relative+.005 absolute tolerance). `test_mid_depth_pore_pressure_matches_terzaghi_series` and `test_settlement_matches_degree_of_consolidation_identity` check outputs separately. `test_time_factor_similarity_controls_consolidation_degree_and_normalized_pore_pressure` probes nondimensional scaling. `test_fidelity_arguments_do_not_change_analytical_response` establishes that this is no longer a CN numerical solver. Cross-validation long-/early-time and load-scaling tests are analytical controls.

Test execution on 2026-09-17: the targeted test file named above passed all 19 collected tests in a combined 10-file run (163 tests total, exit code 0). Cross-validation functions were inspected but not rerun in this pass. The public `docs/CASE_EVIDENCE_INDEX.md` also summarizes test execution; the present note records its own scope and result. Metadata/string tests check documentation contracts, not physical accuracy. Input validation and finite-output tests protect the computational contract, not real-world predictive validity.

## Limits and documentation discrepancies

CSV:P2 analytical control, not external site-specific geotechnical validation. Fixed mv and hydraulic conductivity, small-strain 1 D settlement and single drainage are assumptions. At mvΔσ up to 1 the nominal final strain reaches 100%; the algebraic benchmark can therefore exceed small-strain physical applicability. No field/oedometer fitting, secondary compression or coupled 3 D poromechanics is supplied. The “S-shaped U vs Tv” description is not a calibrated empirical observation. Historical CN wording must not imply a current time-stepping method.

**中文证据边界：** 本页区分代码实际方程、可追溯出版来源和本地测试范围。公式回归、解析极限或数值自洽不等于真实试验校准；未读取的原书页码和未验证的经验系数均不视为已证实。既有 P1/P2 状态未升级，代码及归档未改动。
