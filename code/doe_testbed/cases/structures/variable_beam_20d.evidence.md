# Evidence note: variable_beam_20d

Audit date: 2026-09-17. This note documents the current implementation; it does not modify or certify the archived response data.

- case_id: `variable_beam_20d`
- source_module: `doe_testbed/cases/structures/variable_beam_nd.py`
- targeted_testfile: `tests/test_variable_beam_physics.py`
- cross_validation_testfile: `tests/_cross_validation/variable_beam_20d.py`
- status_source: `docs/case_validation_status.csv`; risk `P1`; disposition: **accepted after resweep**.

## Implemented model and parameters

This registered case has exactly 20 independent heights h 1…h 20, each 0.02–0.15 m, and 20 equal-length segments of a 2 m clamped-free beam. Fixed width .1 m, E 200 GPa, density 7850 kg/m³. Each segment uses cubic Hermite stiffness EI/le³ and consistent mass ρAle/420 matrices, I=bh³/12, A=bh. Root displacement/rotation are removed, leaving 40 active DOFs. scipy.linalg.eigh extracts two lowest eigenvalues; f=√λ/(2π), mass=ρb(L/20)Σhi. Tiny negative eigenvalues within a matrix-scale tolerance are replaced by zero; substantial negatives or failed solves raise.

## Traceable external evidence

[TU Graz university cantilever derivation](https://lampz.tugraz.at/~hadley/memm/mechanics/cbeam.php), checked 2026-09-17, gives cosh(β)cos(β)+1=0, roots 1.875,4.694 and ω=√(EI/μ)k². This directly supports the uniform-beam control, not arbitrary stepped-beam exactness. [Cambridge Petyt 2 nd edition](https://www.cambridge.org/core/books/introduction-to-finite-element-vibration-analysis/4345EDA6046AF3D6BEC8FD29F00AD646), previously verified 2010/ISBN 9780521191609, is relevant FEM background. The cross-test §4.3 locator is unresolved (chapter 4 is in-plane plates). Cook chapter 11 and Inman locators were not inspected.

Supplementary internal audit history (not required to interpret this note; may be absent from the public source package): `paper-reviews/communications-engineering/2026-09-16/case_sources_01_15.md` and `case_sources_31_43.md`. Bibliographic existence and equation-level verification are distinct. No unstated book pages, figure numbers or measured reference data are claimed here.

## Local tests and what they establish

`test_uniform_beam_matches_euler_bernoulli_cantilever_frequencies` is explicitly parameterized for this 20-segment class: at uniform h=.08 m it requires first-frequency relative error<.001 and second-frequency error<0.00002. `test_fem_beam_outputs_remain_ordered_and_finite_after_eigenvalue_sanitization` and negative-eigenvalue tests check numerical contracts. This case's separate cross-validation file checks common-height frequency scaling and linear total mass; these are not independent stepped-geometry reference data.

Test execution on 2026-09-17: the targeted test file named above passed all 15 collected tests in a combined 10-file run (163 tests total, exit code 0). The 15-test file covers all three registered beam dimensions; its count must not be added three times. Cross-validation functions were inspected but not rerun in this pass. The public `docs/CASE_EVIDENCE_INDEX.md` also summarizes test execution; the present note records its own scope and result. Metadata/string tests check documentation contracts, not physical accuracy. Input validation and finite-output tests protect the computational contract, not real-world predictive validity.

## Limits and documentation discrepancies

CSV:P1 accepted after resweep is a project disposition, not experimental accreditation. One element per segment gives a discrete approximation; a uniform-beam control does not bound arbitrary 20-segment errors. Increasing a local height increases both stiffness and mass, so global “smooth_monotone” metadata is not a proof that every local derivative of every frequency is positive. The claimed root-dominant Sobol values are benchmark metadata, not literature-derived universal constants. No shear deformation, rotary inertia, damping, stress recovery or measured modal calibration is included.

**中文证据边界：** 本页区分代码实际方程、可追溯出版来源和本地测试范围。公式回归、解析极限或数值自洽不等于真实试验校准；未读取的原书页码和未验证的经验系数均不视为已证实。既有 P1/P2 状态未升级，代码及归档未改动。
