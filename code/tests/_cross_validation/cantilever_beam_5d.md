# Cross-Validation: `cantilever_beam_5d`

## 1. Reference Source

| Item | Detail |
|---|---|
| Primary reference | `\cite{Shigley2020}` Table A-9, Case 1 (cantilever, point load at free end). |
| Secondary reference | `\cite{Roark2002}` Table 8.1, Case 1a. |
| Formula class | Analytical control (Euler–Bernoulli plus shear-deflection regression) |
| Validation scope | Formula-regression control for the implemented ideal rectangular cantilever model; not calibrated external validation of arbitrary beam hardware or nonlinear structural response. |

## 2. Test Configuration

**Governing equations (both references agree):**

$$
\sigma_{\max} = \frac{6FL}{bh^2}, \quad
\delta_{\max} = \frac{FL^3}{3EI_z}, \quad
\tau_{\max} = \frac{3F}{2bh}, \quad
I_z = \frac{bh^3}{12}
$$

**Test vectors (physical units):**

| # | F (kN) | L (m) | b (m) | h (m) | E (GPa) | Note |
|---|---|---|---|---|---|---|
| 1 | 10 | 1.5 | 0.10 | 0.20 | 200 | Mid-range |
| 2 | 25 | 2.0 | 0.15 | 0.30 | 120 | Large cross-section |
| 3 | 40 | 2.5 | 0.06 | 0.12 | 70  | High-stress corner |

**Environment:** Python 3.11, numpy, doe_testbed v0.3.x.

## 3. Results

### Test vector 1: F = 10 kN, L = 1.5 m, b = 0.10 m, h = 0.20 m, E = 200 GPa

| Output | Expected (hand calc) | Computed | Relative error |
|---|---|---|---|
| σ_max (MPa) | 22.500 | 22.500 | < 1e-14 |
| δ_max (mm)  | 0.84375 | 0.84375 | < 1e-14 |
| τ_max (MPa) | 0.750 | 0.750 | < 1e-14 |

### Test vector 2: F = 25 kN, L = 2.0 m, b = 0.15 m, h = 0.30 m, E = 120 GPa

| Output | Expected | Computed | Relative error |
|---|---|---|---|
| σ_max (MPa) | 22.222 | 22.222 | < 1e-14 |
| δ_max (mm)  | 1.6461 | 1.6461 | < 1e-14 |
| τ_max (MPa) | 0.8333 | 0.8333 | < 1e-14 |

### Test vector 3: F = 40 kN, L = 2.5 m, b = 0.06 m, h = 0.12 m, E = 70 GPa

| Output | Expected | Computed | Relative error |
|---|---|---|---|
| σ_max (MPa) | 578.70 | 578.70 | < 1e-14 |
| δ_max (mm)  | 172.83 | 172.83 | < 1e-14 |
| τ_max (MPa) | 4.1667 | 4.1667 | < 1e-14 |

**Conclusion:** All outputs match the implemented closed-form beam equations to machine precision. This is a formula-regression control for the declared ideal beam model, not a case-level external validation claim.
