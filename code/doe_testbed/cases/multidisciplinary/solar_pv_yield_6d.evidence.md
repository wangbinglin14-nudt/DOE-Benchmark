# Evidence companion: `solar_pv_yield_6d`

Audit date: 2026-09-17. This file documents the current implementation; it does not change the response surface or upgrade validation status.

## Implementation and provenance

- Source: [solar_pv_yield_6d.py](solar_pv_yield_6d.py); SHA-256 `de4f082afef77d9cb1138e9c2c66d531b22a3058e68d5e1adf0fff574bdf52d1`.
- Model: Fixed-parameter annual per-kWp photovoltaic yield surrogate.
- Registry: `docs/case_validation_status.csv` records `P2`, `plausibility only`, historical external validation-note folder `NONE`. The existing cross-validation Markdown is an internal review companion, not an external validation dataset.
- Public `evaluate` uses normalized inputs through BaseCase; the table below is the physical-input contract.

## Inputs and outputs

| Input in column order | Unit | Physical interval | Scale |
|---|---|---|---|
| `GHI_annual` | kWh/m² | [800.0, 2200.0] | linear |
| `tilt` | deg | [0.0, 60.0] | linear |
| `azimuth_dev` | deg | [-30.0, 30.0] | linear |
| `eta_STC` | - | [0.14, 0.25] | linear |
| `beta_T` | %/°C | [-0.5, -0.2] | linear |
| `PR` | - | [0.7, 0.9] | linear |

Outputs, in column order: `E_annual`, `CF`, `T_cell_avg`.

## Equations, constants and implementation choices

f_tilt=.8*1.2*cos(tilt-35deg)^2+.15*(1+cos(tilt))/2+.05*.2*(1-cos(tilt))/2; G_eff=GHI*f_tilt*cos(azimuth_dev)^2; Tcell=25+[20-60*(eta-.20)]*G_eff/1400; Tcorr=1+(beta_T/100)*(Tcell-25); Eannual=G_eff*Tcorr*PR/1kWm^-2; CF=Eannual/8760. eta cancels the per-kWp nominal normalization and affects output only via Tcell. East/west deviations are symmetric by construction.

All named constants above are taken from executable source. A constant is not externally calibrated merely because it has a plausible physical scale.

## Backend and failure contract

NumPy only; no pvlib, PVWatts API, SAM, weather files or measured-plant data. Invalid physical inputs and malformed/nonphysical outputs are checked in the case; see exact validator and response checks in source. Backend package versions and archived source fingerprints must be taken from the production environment manifest; this documentation does not assert equivalence between current and archived versions.

## Source evidence and locator status

| Source | What was verified / permitted use |
|---|---|
| [Sandia PVPMC annual yield](https://pvpmc.sandia.gov/modeling-guide/5-ac-system-output/pv-performance-metrics/annual-yield/) | opened 2026-09-17; metric definition only, not empirical thermal constants. |
| [Duffie/Beckman4th edition2013](https://onlinelibrary.wiley.com/doi/book/10.1002/9781118671603) | publisher metadata verified2026-09-16; ISBN9780470873663; specific §2/§23 equations not full-text verified. |

No unlisted textbook page, equation number, or table has been independently checked in full text during this pass. Official metadata, a conceptual reference and an external quantitative validation comparison are different evidence levels.

## Tests and what they establish

The existing ten-module targeted physics suite was executed on 2026-09-17 with `python -m pytest ... -q --tb=short` and exited 0. This is a local regression result, not a production resweep or experimental validation. For this case:

| Executable test | Evidence scope |
|---|---|
| [test_per_kwp_yield_efficiency_effect_is_only_temperature_coupled](../../../tests/test_solar_pv_yield_physics.py#L796) | At eta.14/.25 andGHI800/2200, cooler high-efficiency modules produce small gain(<2%/<5%), not direct eta-ratio scaling. |
| [test_annual_azimuth_deviation_factor_is_symmetric_by_construction](../../../tests/test_solar_pv_yield_physics.py#L762) | Pins even angular function, not measured east/west equivalence. |
| [test_capacity_factor_matches_specific_yield_definition](../../../tests/test_solar_pv_yield_physics.py#L820) | Pins CF fraction=yield/8760. |
| [test_performance_ratio_is_linear_loss_multiplier](../../../tests/test_solar_pv_yield_physics.py#L670) | Pins PR proportionality. |

Additional tests in the same module check metadata wording, documentation presence, domain corners and failure contracts. Text-string assertions establish that a warning is present; they do not establish the truth of the referenced physical claim. The companion [cross-validation note](../../../tests/_cross_validation/solar_pv_yield_6d.md) and [cross-validation script](../../../tests/_cross_validation/solar_pv_yield_6d.py) were inspected as existing evidence; their historical numeric assertions are not a fresh external-data comparison.

## Simplifications and unsupported claims

The annual irradiation-to-temperature coupling is empirical; it is not a physical hourly heat balance or a standard NOCT calculation. No solar-position/weather series, inverter clipping, module IV/diode model, shading, bifacial effects, uncertainty or bankable site assessment.

## Legacy comments and remaining checks

Code comment calls transposition "Liu–Jordan isotropic sky" but only the diffuse/ground shapes resemble that model; fixed beam weights and cos-squared annual factor are additional surrogate assumptions. "PVWatts-style" is not tool equivalence. Comment temperature-change magnitude is heuristic, not a domain-wide6degree prediction.

## 中文证据边界

本说明按实际代码列出方程、参数及测试范围，不改变源码或归档。当前仍为 P2：压力容器属于解析公式控制，其余属于物理合理性检查；均不能据此声称已获实验、高保真仿真或行业软件校准。未取得原书全文的页码、公式号、表号不标为“已核验”。局部单元测试通过不等于覆盖所有输入、网格收敛或重跑正式 sweep。模型或后端若改变，须按项目协议重新核验并重跑相应结果。
