# Evidence note: tuned_mass_damper_3d

Audit date: 2026-09-17. This note documents the current implementation; it does not modify or certify the archived response data.

- case_id: `tuned_mass_damper_3d`
- source_module: `doe_testbed/cases/structures/tuned_mass_damper_3d.py`
- targeted_testfile: `tests/test_tuned_mass_damper_physics.py`
- cross_validation_testfile: `tests/_cross_validation/tuned_mass_damper_3d.py`
- status_source: `docs/case_validation_status.csv`; risk `P1`; disposition: **accepted after resweep**.

## Implemented model and parameters

Undamped primary oscillator plus damped absorber under harmonic force. μ=0.005–0.10, f=0.70–1.30, ζ=0.01–0.30. A=f²−r², B=2ζfr, C=(f²−r²)(1−r²)−μr²f², D=2ζfr(1−r²−μr²). H=√(A²+B²)/max(√(C²+D²),1e−15). The frequency grid has 160 points [0.01,0.5),1200 [0.5,1.5),320 [1.5,4]. Hmax is capped 100; force ratio=max μr²√(f⁴+B²)/den capped 50. BW is last minus first r with H>capped_Hmax/√2, or zero if the region touches either grid end.

## Traceable external evidence

[MIT's official reading list](https://ocw.mit.edu/courses/16-07-dynamics-fall-2009/pages/readings/) confirms Den Hartog, Mechanical Vibrations, fourth edition,1956. [Asami et al. Crossref](https://api.crossref.org/works/10.1115/1.1456458) confirms the analytical absorber-optimization article 124(2),284–295(2002); metadata was saved in the previous audit. Neither Den Hartog §3.3 nor the source's Rana/Soong equation 1 a has been read here, so exact equation-level attribution remains provisional.

Supplementary internal audit history (not required to interpret this note; may be absent from the public source package): `paper-reviews/communications-engineering/2026-09-16/case_sources_01_15.md` and `case_sources_31_43.md`. Bibliographic existence and equation-level verification are distinct. No unstated book pages, figure numbers or measured reference data are claimed here.

## Local tests and what they establish

`test_den_hartog_optimum_remains_resolved_with_refined_sweep` sets μ=.05, f=1/(1+μ), ζ=√[3μ/(8(1+μ))] and checks Hmax within 2% of√(1+2/μ). Cross-validation repeats equal-peak controls at μ=.02,.05,.10. `test_bandwidth_returns_zero_when_half_power_region_is_not_closed` pins edge handling. These validate sampled approximate control points, not continuous-frequency exact maxima or installed absorbers.

Test execution on 2026-09-17: the targeted test file named above passed all 9 collected tests in a combined 10-file run (163 tests total, exit code 0). Cross-validation functions were inspected but not rerun in this pass. The public `docs/CASE_EVIDENCE_INDEX.md` also summarizes test execution; the present note records its own scope and result. Metadata/string tests check documentation contracts, not physical accuracy. Input validation and finite-output tests protect the computational contract, not real-world predictive validity.

## Limits and documentation discrepancies

The docstring output table writes BW for H<threshold; the implementation uses H>threshold. For separated peaks the returned interval spans gaps and is not necessarily the sum of half-power band lengths. Clipping and finite frequency sampling affect extreme peaks and force ratios. Equal-peak controls should not be described as an exact all-frequency H∞ solution. No primary structural damping, nonlinear devices, building calibration or base-excitation validation is included.

**中文证据边界：** 本页区分代码实际方程、可追溯出版来源和本地测试范围。公式回归、解析极限或数值自洽不等于真实试验校准；未读取的原书页码和未验证的经验系数均不视为已证实。既有 P1/P2 状态未升级，代码及归档未改动。
