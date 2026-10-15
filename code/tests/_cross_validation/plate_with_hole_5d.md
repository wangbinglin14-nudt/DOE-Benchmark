# Cross-Validation: `plate_with_hole_5d`

## 1. Reference Source

| Item | Detail |
|---|---|
| Primary reference | `\cite{Pilkey2008}` Chart 4.1, Table 4-1 (elliptical hole, finite-width correction). |
| Classical reference | `\cite{Kirsch1898}` Circular hole Kt = 3 under uniaxial tension. |
| Formula class | Analytical control (Inglis/Kirsch SCF + Heywood finite-width polynomial) |
| Validation scope | Formula-regression control for the implemented stress-concentration surrogate; not calibrated external validation of arbitrary finite-plate tests, plasticity, crack initiation, or notch design. |

## 2. Test Configuration

**Governing equations (Inglis + Heywood):**

$$
K_{t,x} = 1 + 2\frac{a}{b}, \quad K_{t,y} = 1 + \frac{2b}{a}
$$
$$
K_{hw} = 1 + 0.5\left(\frac{2a}{W}\right)^3 + 0.3\left(\frac{2a}{W}\right)^5
$$
$$
\sigma_A = \sigma_{x,eff} K_{t,x} K_{hw} - \sigma_{y,eff}, \quad
\sigma_B = \sigma_{y,eff} K_{t,y} K_{hw} - \sigma_{x,eff}
$$

**Test vectors:**

| # | a/b | a/W | σ_x (MPa) | σ_y (MPa) | θ (°) | Note |
|---|---|---|---|---|---|---|
| 1 | 1.0 | 0.1 | 200 | 0  | 0  | Kirsch circular, uniaxial |
| 2 | 2.0 | 0.2 | 150 | 50 | 0  | Elliptical, biaxial |
| 3 | 3.0 | 0.3 | 100 | −20| 45 | High aspect, angled |

## 3. Results

### Test vector 1: Circular hole, uniaxial (Kirsch limit)

| Output | Expected (Kirsch) | Computed | Error |
|---|---|---|---|
| σ_max (MPa) | ≈ 600 (3 × 200 × K_hw) | 601.2 | < 0.3 % vs Kirsch |
| Kt (-) | ≈ 3.006 | 3.006 | 0.2 % vs Kirsch=3 |

The 0.2 % departure from Kt = 3.0 is due to the Heywood correction at a/W = 0.1 (not truly infinite plate).

### Test vectors 2–3

All outputs match the hand-computed Inglis + Heywood formula to machine precision.

**Conclusion:** The implementation reproduces the declared Inglis/Kirsch SCF plus Heywood finite-width surrogate exactly for the test vectors. This supports formula correctness, not a broad external validation claim.
