# Evidence note: ten_bar_truss_10d

Audit date: 2026-09-17. This note documents the current implementation; it does not modify or certify the archived response data.

- case_id: `ten_bar_truss_10d`
- source_module: `doe_testbed/cases/structures/ten_bar_truss_10d.py`
- targeted_testfile: `tests/test_ten_bar_truss_physics.py`
- cross_validation_testfile: `tests/_cross_validation/ten_bar_truss_10d.py`
- status_source: `docs/case_validation_status.csv`; risk `P1`; disposition: **accepted after resweep**.

## Implemented model and parameters

Direct-stiffness linear-elastic axial bars: ke=(EA/L) vvᵀ with v=(cx, cy,−cx,−cy); solve Kff uf=Ff after removing support DOFs. E=68.9 GPa, density=2700 kg/m³; ten independent areas 0.645–25.8 cm². Seven nodes in metres: (0,0),(9.144,0),(18.288,0),(0,9.144),(9.144,9.144),(18.288,9.144),(27.432,0). Zero-based bars: (0,1),(1,2),(2,6),(3,4),(4,5),(0,4),(1,4),(1,5),(2,5),(3,1). Both displacement components fixed at nodes 3 and 6; −200 kN vertical at nodes 1 and 2. Returns maxima of absolute free-node x/y displacements (mm) and ΣρAiLi (kg), not stresses.

## Traceable external evidence

[Elsevier Arora 2 nd edition](https://shop.elsevier.com/books/introduction-to-optimum-design/arora/978-0-12-064155-0) and [Springer Haftka/Gürdal 3 rd edition](https://link.springer.com/book/10.1007/978-94-011-2550-5) establish real optimization references, reused from the 2026-09-16 audit. Their claimed §§5.9/7.9 were not read. Haftka hardcover ISBN is 9780792315049. No external source has been verified to contain this exact seven-node connectivity, support pattern and loading.

Supplementary internal audit history (not required to interpret this note; may be absent from the public source package): `paper-reviews/communications-engineering/2026-09-16/case_sources_01_15.md` and `case_sources_31_43.md`. Bibliographic existence and equation-level verification are distinct. No unstated book pages, figure numbers or measured reference data are claimed here.

## Local tests and what they establish

`test_fixed_support_dofs_are_removed_and_reconstructed_as_zero` checks support reconstruction. `test_constrained_solve_matches_penalty_response_without_ill_conditioning` compares two numerical treatments of the same assembled problem, rtol 1e-12, and checks reduced-system conditioning. Cross-validation `test_displacement_scales_inversely_with_area` and `test_mass_linear_in_area` test uniform-area scaling. These do not independently verify literature geometry or measured deformation.

Test execution on 2026-09-17: the targeted test file named above passed all 9 collected tests in a combined 10-file run (163 tests total, exit code 0). Cross-validation functions were inspected but not rerun in this pass. The public `docs/CASE_EVIDENCE_INDEX.md` also summarizes test execution; the present note records its own scope and result. Metadata/string tests check documentation contracts, not physical accuracy. Input validation and finite-output tests protect the computational contract, not real-world predictive validity.

## Limits and documentation discrepancies

The source calls the case a “cantilever” and “standard ten-bar” problem, but its two separated pin supports and seven-node layout must be identified explicitly instead of presumed identical to a canonical six-node ten-bar optimization benchmark. The “6 active nodes + 1 support” comment conflicts with two supported nodes. Linear equilibrium does not imply displacement is linear in design areas. No buckling, yielding, joint flexibility or experimental truss calibration is represented.

**中文证据边界：** 本页区分代码实际方程、可追溯出版来源和本地测试范围。公式回归、解析极限或数值自洽不等于真实试验校准；未读取的原书页码和未验证的经验系数均不视为已证实。既有 P1/P2 状态未升级，代码及归档未改动。
