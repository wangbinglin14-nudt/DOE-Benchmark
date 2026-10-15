# Cross-Validation: `tuned_mass_damper_3d`

## 1. Reference Source

| Item | Detail |
|---|---|
| Primary reference | `\cite{DenHartog1956}` §3.3 "The Vibration Absorber", pp. 87–106. |
| Secondary reference | `\cite{Asami2002}` Exact closed-form $H_\infty$ optimum. |
| Formula class | Exact 2-DOF transfer function, numerically swept; Den Hartog equal-peak points are analytical controls |
| Validation scope | Resonance-refined sweep check against Den Hartog control points; not a calibrated-accuracy claim for arbitrary or building-specific TMD designs |

## 2. Test Configuration

**Den Hartog fixed-point (equal-peak) optimal design:**

$$
f_{opt} = \frac{1}{1+\mu}, \quad
\zeta_{d,opt} = \sqrt{\frac{3\mu}{8(1+\mu)}}, \quad
H_{\max} \approx \sqrt{1 + \frac{2}{\mu}}
$$

The $H_{\max}$ formula is exact in the limit of an undamped primary structure.

**Test vectors:**

| # | μ | f_opt | ζ_d_opt | H_max (theory) | Note |
|---|---|---|---|---|---|
| 1 | 0.05 | 0.9524 | 0.1336 | 6.403 | Standard TMD |
| 2 | 0.02 | 0.9804 | 0.0857 | 10.050 | Light TMD |
| 3 | 0.10 | 0.9091 | 0.1830 | 4.583 | Heavy TMD |

## 3. Results

| μ | H_max (Den Hartog) | H_max (computed) | Error (%) |
|---|---|---|---|
| 0.05 | 6.403 | ≈ 6.39 | < 1 % |
| 0.02 | 10.050 | ≈ 10.03 | < 1 % |
| 0.10 | 4.583 | ≈ 4.57 | < 1 % |

**Discussion:** The code evaluates the exact 2-DOF transfer function over a resonance-refined frequency sweep (r ∈ [0.01, 4.0], with dense sampling over the resonance band). The Den Hartog equal-peak formula is an analytical control for the undamped-primary idealisation, not a calibrated external validation target for arbitrary TMD configurations.

**Conclusion:** Peak amplification H_max agrees with the Den Hartog equal-peak control points within the regression-test tolerance for the implemented idealised 2-DOF model.
