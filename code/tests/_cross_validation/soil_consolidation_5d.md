# Cross-Validation: `soil_consolidation_5d`

## 1. Reference Source

| Item | Detail |
|---|---|
| Primary reference | Terzaghi one-dimensional consolidation theory and standard geotechnical textbook Tv-U relations. |
| Formula class | Analytical control (single-drainage Terzaghi Fourier-series solution) |
| Evidence tier | P2 analytical-control |
| Validation scope | Formula-regression control for the implemented one-dimensional single-drainage analytical surrogate; not calibrated site-specific geotechnical settlement validation. |

Public source anchors:

- Itasca FLAC3D one-dimensional consolidation verification example:
  https://docs.itascacg.com/itasca940/flac3d/zone/test3d/Fluid/1DConsolidation/1dconsolidation.html
- MOLE Terzaghi one-dimensional consolidation analytical benchmark:
  https://mole-docs.readthedocs.io/en/main/examples/Parabolic/1D/Terzaghi1D.html
- OpenGeoSys hydro-mechanics benchmark overview:
  https://www.opengeosys.org/docs/benchmarks/hydro-mechanics/
- OpenGeoSys hydro-mechanics benchmark overview (stable docs path):
  https://www.opengeosys.org/stable/docs/benchmarks/hydro-mechanics/
- OpenGeoSys ConsolidationBenchmark project documentation:
  https://doxygen.opengeosys.org/6.5.6/d7/df1/ogs_ctest_prj__HydroMechanics__StaggeredScheme__ConsolidationBenchmark__consolidation_benchmark__prj.html

## 2. Implemented Equations

The case evaluates the Terzaghi excess-pore-pressure diffusion equation for a
single-drainage layer:

$$
\frac{\partial u_e}{\partial t} = c_v \frac{\partial^2 u_e}{\partial z^2},
\quad c_v = \frac{k}{m_v \gamma_w},
\quad T_v = \frac{c_v t}{H_{dr}^2}.
$$

The average degree of consolidation is evaluated with the Fourier series

$$
U(T_v) = 1 - \sum_{m=0}^{\infty} \frac{2}{M_m^2}\exp(-M_m^2 T_v),
\quad M_m = \frac{\pi(2m+1)}{2}.
$$

Settlement follows

$$
S(t)=U(T_v)m_v \Delta\sigma H_{dr}.
$$

## 3. Checked Test Controls

The Python cross-validation file verifies:

- long-time convergence to full consolidation,
- partial-consolidation agreement with the analytical Tv-U curve,
- settlement monotonicity with surcharge load,
- settlement identity \(S(t)=U(t)m_v\Delta\sigma H_{dr}\),
- mid-depth pore-pressure agreement with the Fourier-series expression,
- time-factor similarity: holding \(T_v=c_vt/H_{dr}^2\) fixed holds
  \(U(T_v)\) and \(u_{mid}/\Delta\sigma\) fixed, while settlement scales with
  \(H_{dr}\) through \(S_{final}=m_v\Delta\sigma H_{dr}\),
- invalid private physical-input rejection before archive rows can be produced.

## 4. Claim Boundary

Evidence tier remains P2 analytical-control. The public Itasca and MOLE pages
are used as independent Terzaghi analytical benchmark anchors; the Itasca page
also serves as Biot-theory/Darcy-flow boundary context for an elastic matrix,
drained boundary, impervious base, and numerical-vs-analytical pore-pressure
comparison. OpenGeoSys material is used as boundary evidence for coupled
hydro-mechanical workflows. These sources do not validate field-specific soil
parameters, double-drainage selection, radial drains, layered or anisotropic
deposits, stress-dependent properties, finite-strain effects, secondary
compression, construction staging, unsaturated flow, laboratory or field
oedometer fitting, or 2-D/3-D coupled Biot poromechanics. The case must not be
described as external site-specific geotechnical validation.

The implementation rejects finite but non-physical or out-of-contract
`S_t`/`U_t`/`u_mid` responses before archive rows can be produced. This is a
row-integrity guard only; it does not change valid analytical Terzaghi
responses in the declared input domain.
