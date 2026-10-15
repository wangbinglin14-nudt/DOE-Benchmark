# Cross-Validation: `pressure_vessel_4d`

## 1. Reference Source

| Item | Detail |
|---|---|
| Primary reference | `\cite{Shigley2020}` §3-14, Thin-walled pressure vessels. |
| Secondary reference | `\cite{Boresi2003}` §4.5 (von Mises criterion). |
| Formula class | Analytical control (thin-wall membrane theory + von Mises) |
| Evidence tier | P2 analytical-control |
| Validation scope | Formula-regression control for the implemented strength-only thin-wall surrogate; not a full ASME BPVC, thick-wall, radial-stress, end-cap/support, nozzle/opening, external-pressure buckling, weld, corrosion, fatigue, fracture, hydrostatic-test, inspection, or fabrication design validation. |

Public source anchors:

- ASME BPVC Section VIII, Division 1 product page:
  https://www.asme.org/codes-standards/find-codes-standards/bpvc-viii-1-bpvc-section-viii-rules-construction-pressure-vessels-division-1
- ASME Boiler and Pressure Vessel Certification overview:
  https://www.asme.org/certification-accreditation/Boiler-and-Pressure-Vessel-Certification
- NASA/TM-2015-218216 thin pressure-vessel pipe formulation:
  https://ntrs.nasa.gov/api/citations/20160000456/downloads/20160000456.pdf

## 2. Test Configuration

**Governing equations:**

$$
\sigma_{VM} = \frac{PR}{t} \cdot \frac{\sqrt{3}}{2}, \quad
FS = \frac{\sigma_y}{\sigma_{VM}}, \quad
m/L = \rho_{steel} \cdot \pi \left[(R+t)^2 - R^2\right]
$$

with $\rho_{steel} = 7850$ kg/m³.

**Test vectors:**

| # | P (MPa) | R (m) | t (m) | σ_y (MPa) | Note |
|---|---|---|---|---|---|
| 1 | 10 | 0.5 | 0.02 | 350 | Standard dimensions |
| 2 | 5  | 1.0 | 0.03 | 400 | Large radius |
| 3 | 15 | 1.5 | 0.01 | 250 | Thin wall, high pressure |

## 3. Results

### Test vector 1: P = 10 MPa, R = 0.5 m, t = 0.02 m, σ_y = 350 MPa

| Output | Expected | Computed | Relative error |
|---|---|---|---|
| FS (-)        | 1.6166 | 1.6166 | < 1e-14 |
| W_per_L (kg/m)| 503.07 | 503.07 | < 1e-14 |
| σ_VM (MPa)   | 216.51 | 216.51 | < 1e-14 |

### Test vector 2: P = 5 MPa, R = 1.0 m, t = 0.03 m, σ_y = 400 MPa

| Output | Expected | Computed | Relative error |
|---|---|---|---|
| FS (-)        | 2.7713 | 2.7713 | < 1e-14 |
| W_per_L (kg/m)| 1501.5 | 1501.5 | < 1e-14 |
| σ_VM (MPa)   | 144.34 | 144.34 | < 1e-14 |

### Test vector 3: P = 15 MPa, R = 1.5 m, t = 0.01 m, σ_y = 250 MPa

| Output | Expected | Computed | Relative error |
|---|---|---|---|
| FS (-)        | 0.1283 | 0.1283 | < 1e-14 |
| W_per_L (kg/m)| 739.1 | 739.1 | < 1e-14 |
| σ_VM (MPa)   | 1948.6 | 1948.6 | < 1e-14 |

**Conclusion:** All outputs match the implemented thin-wall + von Mises formula to machine precision. This supports formula correctness for the declared strength-only surrogate. Evidence tier remains P2 analytical-control: the case is not externally validated ASME BPVC design-code, thick-wall, local-stress, buckling, fatigue/fracture, fabrication, certification, pressure-relief, or inspection validation.
