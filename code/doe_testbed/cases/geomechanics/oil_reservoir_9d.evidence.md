# Evidence note: oil_reservoir_9d

Audit date: 2026-09-17. This note documents the current implementation; it does not modify or certify the archived response data.

- case_id: `oil_reservoir_9d`
- source_module: `doe_testbed/cases/geomechanics/oil_reservoir_9d.py`
- targeted_testfile: `tests/test_oil_reservoir_physics.py`
- cross_validation_testfile: `tests/_cross_validation/oil_reservoir_9d.py`
- status_source: `docs/case_validation_status.csv`; risk `P2`; disposition: **plausibility only**.

## Implemented model and parameters

Nine 3×3 row-major block permeabilities 1–1000 mD;1 m×1 m domain, unit-depth interpretation, viscosity .001 Pa·s, pleft 10 MPa, pright 1 MPa, closed top/bottom. Solve ∇·[(K/μ)∇p]=0 by nodal finite differences with harmonic interfaces. Default 18×18 nodes (6/block), high 72×72. Internal vertical-face flux magnitudes are trapezoid-integrated in y and averaged across faces. Q_total is pressure-normalized conductance m³/(day·MPa), not raw production rate. Central pressure differences give dp_max; equivalent permeability is recovered from conductance, rejecting excursions beyond block-K extrema except roundoff.

## Traceable external evidence

[USGS Circular 1224 groundwater-flow explanation](https://pubs.usgs.gov/circ/2002/circ1224/html/understanding.html) provides official Darcy-law context (indexed content rechecked 2026-09-17). [SPE 10 DOI](https://doi.org/10.2118/72469-PA), MRST and OPM links in the source are contextual comparisons, not reference runs performed by this case. Previous audit verified Aziz/Settari 1979's existence but not its chapter 3 equations, and corrected its ISBN to 9780853347873; Bear edition mixing was also flagged. No independently solved reservoir dataset is imported here.

Supplementary internal audit history (not required to interpret this note; may be absent from the public source package): `paper-reviews/communications-engineering/2026-09-16/case_sources_01_15.md` and `case_sources_31_43.md`. Bibliographic existence and equation-level verification are distinct. No unstated book pages, figure numbers or measured reference data are claimed here.

## Local tests and what they establish

`test_uniform_permeability_matches_absolute_darcy_scale` checks the absolute scale. Column-layer/row-layer tests compare harmonic/arithmetic effective permeability:3% default,1% high-grid tolerances for the chosen layered examples. `test_internal_face_flux_diagnostics_pin_mass_balance_convergence` requires relative internal-face spread<4% at 18×18 and<1% at 72×72 for one specified heterogeneous field. Mirror and common-K-scaling tests probe operator structure. These are numerical controls, not an archive-wide convergence certificate.

Test execution on 2026-09-17: the targeted test file named above passed all 35 collected tests in a combined 10-file run (163 tests total, exit code 0). Cross-validation functions were inspected but not rerun in this pass. The public `docs/CASE_EVIDENCE_INDEX.md` also summarizes test execution; the present note records its own scope and result. Metadata/string tests check documentation contracts, not physical accuracy. Input validation and finite-output tests protect the computational contract, not real-world predictive validity.

## Limits and documentation discrepancies

CSV:P2 plausibility-only. The domain is a small pressure-boundary permeability problem, not well production, recovery or calibrated petroleum forecasting. Averaging flux magnitudes is the chosen response definition and finite-grid spread remains possible. Uniform/selected layered checks cannot certify all strong contrasts. There is no multiphase saturation, well index, storage, history matching or measured field calibration. The docstring phrase “fracture-detection indicator” for dp_max is contextual, not a validated classifier.

**中文证据边界：** 本页区分代码实际方程、可追溯出版来源和本地测试范围。公式回归、解析极限或数值自洽不等于真实试验校准；未读取的原书页码和未验证的经验系数均不视为已证实。既有 P1/P2 状态未升级，代码及归档未改动。
