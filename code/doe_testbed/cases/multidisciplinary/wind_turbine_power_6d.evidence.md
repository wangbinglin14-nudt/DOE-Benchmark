# Evidence companion: `wind_turbine_power_6d`

Audit date: 2026-09-17. This file documents the current implementation; it does not change the response surface or upgrade validation status.

## Implementation and provenance

- Source: [wind_turbine_power_6d.py](wind_turbine_power_6d.py); SHA-256 `a57d91d1283228e39cb46c9484b1384a4d7be1c61fce73faf7658c40984b38ed`.
- Model: Betz-parameterized power curve and closed-form Rayleigh AEP.
- Registry: `docs/case_validation_status.csv` records `P2`, `plausibility only`, historical external validation-note folder `NONE`. The existing cross-validation Markdown is an internal review companion, not an external validation dataset.
- Public `evaluate` uses normalized inputs through BaseCase; the table below is the physical-input contract.

## Inputs and outputs

| Input in column order | Unit | Physical interval | Scale |
|---|---|---|---|
| `R` | m | [30.0, 80.0] | linear |
| `lam_opt` | - | [6.0, 10.0] | linear |
| `c_r` | m | [2.0, 5.0] | linear |
| `theta_0` | deg | [8.0, 20.0] | linear |
| `V_rated` | m/s | [10.0, 14.0] | linear |
| `V_bar` | m/s | [5.0, 12.0] | linear |

Outputs, in column order: `Cp_max`, `P_rated`, `AEP`.

## Equations, constants and implementation choices

rho=1.225kg/m3; B=3; c_avg=.6*c_r; sigma=B*c_avg/(pi*R); Cp=(16/27)*(1-1.5/(B*lambda_opt))*(1-.3sigma); k=.5*rho*pi*R^2*Cp; Prated=k*V_rated^3. P(V)=0 below3m/s or above25m/s, kV^3 between3 andrated, Prated betweenrated and25. Rayleigh scale c=2*Vbar/sqrt(pi),u=(V/c)^2. AEP_GWh=8760/1e9*[k*c^3*Gamma(2.5)*(Pgamma(2.5,u_rated)-Pgamma(2.5,u_cut_in))+Prated*(exp(-u_rated)-exp(-u_cut_out))]. Pgamma is regularized lower incomplete gamma. theta_0 is unused. Probability outside operating interval is not renormalized.

All named constants above are taken from executable source. A constant is not externally calibrated merely because it has a plausible physical scale.

## Backend and failure contract

NumPy plus scipy.special.gamma/gammainc; no OpenFAST/AeroDyn, SAM, wind-resource time series or power-curve database. Invalid physical inputs and malformed/nonphysical outputs are checked in the case; see exact validator and response checks in source. Backend package versions and archived source fingerprints must be taken from the production environment manifest; this documentation does not assert equivalence between current and archived versions.

## Source evidence and locator status

| Source | What was verified / permitted use |
|---|---|
| [Wiley Manwell second edition](https://onlinelibrary.wiley.com/doi/book/10.1002/9781119994367) | verified2026-09-16 publisher copyright2009, ISBN9780470015001; Rayleigh exact page not verified. |
| [Wiley-VCH contents](https://www.wiley-vch.de/en/areas-interest/engineering/wind-energy-explained-978-0-470-01500-1) | verified2026-09-16 contents§3.6 and§3.11 only; not a full-text formula audit. |
| [OpenFAST AeroDyn introduction](https://openfast.readthedocs.io/en/main/source/user/aerodyn/introduction.html) | opened2026-09-17; production-tool boundary, no calibration. |

No unlisted textbook page, equation number, or table has been independently checked in full text during this pass. Official metadata, a conceptual reference and an external quantitative validation comparison are different evidence levels.

## Tests and what they establish

The existing ten-module targeted physics suite was executed on 2026-09-17 with `python -m pytest ... -q --tb=short` and exited 0. This is a local regression result, not a production resweep or experimental validation. For this case:

| Executable test | Evidence scope |
|---|---|
| [test_aep_matches_closed_form_rayleigh_piecewise_power_curve](../../../tests/test_wind_turbine_power_physics.py#L710) | Same closed-form gamma integral at a representative state to1e-12; formula regression, not independent wind data. |
| [test_aep_integration_matches_cut_in_cut_out_zero_power_mask](../../../tests/test_wind_turbine_power_physics.py#L682) | Numerical integration/control of piecewise operating mask. |
| [test_theta0_is_declared_neutral_in_low_order_surrogate](../../../tests/test_wind_turbine_power_physics.py#L805) | Pins inactive pitch coordinate. |
| [test_aep_does_not_exceed_full_year_rated_energy](../../../tests/test_wind_turbine_power_physics.py#L751) | Physical upper bound, not energy-yield validation. |

Additional tests in the same module check metadata wording, documentation presence, domain corners and failure contracts. Text-string assertions establish that a warning is present; they do not establish the truth of the referenced physical claim. The companion [cross-validation note](../../../tests/_cross_validation/wind_turbine_power_6d.md) and [cross-validation script](../../../tests/_cross_validation/wind_turbine_power_6d.py) were inspected as existing evidence; their historical numeric assertions are not a fresh external-data comparison.

## Simplifications and unsupported claims

Algebraic finite-blade factor is not Prandtl BEM iteration. No detailed blade polars, generator efficiency, density variation, wakes, turbulence/shear, measured manufacturer curve, control dynamics, IEC power testing or site/farm calibration.

## Legacy comments and remaining checks

Earlier references.bib Manwell2010 year/locators were inconsistent: second edition is2009; BEM overview§3.6, whereas§3.11 is simplified HAWT performance, not Rayleigh distribution. Do not cite that legacy Rayleigh locator as checked. Cp radius dependence through solidity means rated power is not exactly proportional toR^2 when root chord is fixed.

## 中文证据边界

本说明按实际代码列出方程、参数及测试范围，不改变源码或归档。当前仍为 P2：压力容器属于解析公式控制，其余属于物理合理性检查；均不能据此声称已获实验、高保真仿真或行业软件校准。未取得原书全文的页码、公式号、表号不标为“已核验”。局部单元测试通过不等于覆盖所有输入、网格收敛或重跑正式 sweep。模型或后端若改变，须按项目协议重新核验并重跑相应结果。
