"""
Case G04: Terzaghi one-dimensional consolidation
=================================================

Engineering background
----------------------
When a saturated clay layer is loaded by a surface surcharge, the excess
pore-water pressure u_e gradually dissipates as water drains out, and
the soil skeleton compresses (consolidation settlement).  This is a
fundamental problem in geotechnical engineering — design of foundations,
embankments, and reclaimed land depends critically on predicting how
fast and how much the soil settles.

Terzaghi's 1D consolidation theory models a saturated clay layer of
thickness H_dr (drainage path length) with a coefficient of
consolidation c_v and applied surcharge Δσ.

The implementation is a single-drainage Terzaghi analytical-control
benchmark, not a calibrated site-specific settlement model.  It does not model
single-vs-double-drainage design selection, radial/vertical-drain
consolidation, layered or anisotropic permeability, nonlinear
compressibility, stress-dependent hydraulic conductivity, finite-strain
consolidation, secondary compression/creep, construction staging, lateral
strain, 2-D/3-D coupled Biot poromechanics, poroelastic pressure-displacement coupling,
stress/strain field recovery, unsaturated
flow, field instrumentation interpretation, or laboratory/field oedometer
parameter fitting. Itasca's FLAC3D one-dimensional consolidation verification
example is cited as public equation/boundary context: it frames the benchmark
in Biot theory with an elastic soil matrix, Darcy flow, a drained boundary,
an impervious base, and a numerical-vs-analytical pore-pressure comparison.
It is not calibration data for this compact Terzaghi response surface.
OpenGeoSys hydro-mechanics examples, including the published "Consolidation
benchmark with the staggered scheme", are cited only as omitted-scope context
for coupled THMC/HM process modelling, not as validation data for this
one-dimensional benchmark. The OpenGeoSys project file explicitly uses a
HYDRO_MECHANICS process with staggered pressure/displacement coupling,
fixed-stress stabilization, Backward-Euler time stepping, and displacement,
stress/strain, and velocity outputs; those fields and solver workflows are not
implemented by this analytical Terzaghi response surface.

Governing PDE (parabolic diffusion equation)
--------------------------------------------
  ∂u_e/∂t = c_v · ∂²u_e/∂z²

  c_v = k / (m_v · γ_w)

  where:
    u_e = excess pore pressure (Pa)
    k   = hydraulic conductivity (m/s)
    m_v = coefficient of volume compressibility (1/Pa)
    γ_w = unit weight of water ≈ 9810 N/m³

Boundary conditions:
  u_e(z=0, t) = 0          (permeable top boundary)
  ∂u_e/∂z(z=H_dr, t) = 0  (impermeable bottom boundary — single drainage)
  u_e(z, t=0) = Δσ         (uniform initial excess pore pressure)

Analytical solution (Fourier series):
  u_e(z, t) = Σ_{m=0}^{∞} (2·Δσ/M) · sin(M·z/H_dr) · exp(-M²·T_v)
  where M = π(2m+1)/2, T_v = c_v·t/H_dr²

Average degree of consolidation:
  U(t) = 1 − Σ (2/M²) · exp(-M²·T_v)

Final settlement:
  S_final = m_v · Δσ · H_dr
  S(t) = U(t) · S_final

This implementation evaluates the analytical Fourier-series solution directly
for the headline response.  Earlier Crank–Nicolson finite-difference wording is
retained only as historical context: fixed-step CN was stable but underresolved
the extreme large-time-factor corner of the declared input domain.  The current
solver is therefore a deterministic Terzaghi analytical-control benchmark.

Input variables
---------------
k       Hydraulic conductivity    [1e-10, 1e-7]  m/s   Design  (log-scale)
m_v     Volume compressibility    [1e-7, 5e-6]   1/Pa  Design  (log-scale)
H_dr    Drainage path length      [2.0, 20.0]    m     Design
delta_s Surcharge load            [20.0, 200.0]  kPa   Noise
t_obs   Observation time          [0.5, 50.0]    yr    Design

Output responses
----------------
S_t       Settlement at t_obs      mm   Primary consolidation settlement
U_t       Degree of consolidation  -    Fraction of final settlement achieved
u_mid     Mid-layer pore pressure  kPa  Excess pore pressure at z = H_dr/2

DOE validation value
--------------------
VS-A (space-filling):
    Settlement depends strongly on T_v = c_v·t/H_dr²; the response
    is steep near T_v ~ 0.2 and flat beyond T_v ~ 2, creating
    strong heterogeneity across the input space.

VS-C (surrogate accuracy):
    The S-shaped consolidation curve (U vs T_v) creates nonlinear
    coupling between k, m_v, H_dr, and t_obs.

VS-G (sequential reliability):
    Failure threshold S_t > 100 mm defines an allowable settlement
    limit; the benchmark tests whether sequential reliability methods can
    estimate this implicit 5D failure surface efficiently.

References
----------
1. Terzaghi, K. (1943). Theoretical Soil Mechanics. Wiley. Chapter 12.
2. Das, B. M. (2010). Principles of Geotechnical Engineering, 7th ed.
   Cengage. Chapter 11, §11.6.
3. Craig, R. F. (2004). Craig's Soil Mechanics, 7th ed. Spon Press.
   Chapter 7, §7.2-7.4.
4. Itasca FLAC3D one-dimensional consolidation verification example.
   https://docs.itascacg.com/itasca940/flac3d/zone/test3d/Fluid/1DConsolidation/1dconsolidation.html
5. MOLE Terzaghi one-dimensional consolidation analytical benchmark.
   https://mole-docs.readthedocs.io/en/main/examples/Parabolic/1D/Terzaghi1D.html
6. OpenGTI consolidation overview, used only as public boundary context
   for single-vs-double-drainage choice, drainage-path definitions,
   secondary compression, stress-dependent parameters, anisotropy, staged
   loading, and field/lab parameter limits.
   https://opengti.io/consolidation
7. OpenGeoSys hydro-mechanics benchmarks and ConsolidationBenchmark project
   documentation, used only as coupled-poroelastic omitted-scope context.
   https://www.opengeosys.org/docs/benchmarks/hydro-mechanics/
   https://www.opengeosys.org/stable/docs/benchmarks/hydro-mechanics/
   https://doxygen.opengeosys.org/6.5.6/d7/df1/ogs_ctest_prj__HydroMechanics__StaggeredScheme__ConsolidationBenchmark__consolidation_benchmark__prj.html
"""

import numpy as np

from doe_testbed.base.case import BaseCase
from doe_testbed.base.meta import CaseMeta, InputVariable, OutputVariable
from doe_testbed.registry.registry import register_case

_GAMMA_W = 9810.0  # unit weight of water (N/m³)


@register_case
class SoilConsolidation5D(BaseCase):
    """Terzaghi 1-D consolidation settlement, 5D → 3D.

    Evaluates the single-drainage Terzaghi Fourier-series solution directly,
    producing settlement, degree of consolidation, and mid-layer pore pressure.
    """

    meta = CaseMeta(
        case_id="soil_consolidation_5d",
        name="Terzaghi 1-D consolidation settlement (analytical Fourier series)",
        version="1.0.0",
        domain="geomechanics",
        tags=[
            "consolidation",
            "terzaghi",
            "geotechnical",
            "settlement",
            "pore_pressure",
            "clay",
            "analytical",
            "fourier_series",
            "parabolic",
        ],
        dim_input=5,
        dim_output=3,
        input_variables=[
            InputVariable(
                name="k",
                symbol="k",
                description="Hydraulic conductivity of clay",
                unit="m/s",
                lower=1e-10,
                upper=1e-7,
                log_scale=True,
                distribution="log-uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="m_v",
                symbol="m_v",
                description="Coefficient of volume compressibility",
                unit="1/Pa",
                lower=1e-7,
                upper=5e-6,
                log_scale=True,
                distribution="log-uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="H_dr",
                symbol="H_{dr}",
                description="Drainage path length (single drainage)",
                unit="m",
                lower=2.0,
                upper=20.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
            InputVariable(
                name="delta_s",
                symbol=r"\Delta\sigma",
                description="Applied surcharge load",
                unit="kPa",
                lower=20.0,
                upper=200.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="noise",
            ),
            InputVariable(
                name="t_obs",
                symbol="t_{obs}",
                description="Observation time after loading",
                unit="yr",
                lower=0.5,
                upper=50.0,
                log_scale=False,
                distribution="uniform",
                variable_type="continuous",
                role="design",
            ),
        ],
        output_variables=[
            OutputVariable(
                name="S_t",
                symbol="S(t)",
                description="Settlement at observation time",
                unit="mm",
                is_primary=True,
                expected_range=(0.0, 20000.0),
                failure_threshold=100.0,
                failure_direction="above",
            ),
            OutputVariable(
                name="U_t",
                symbol="U(t)",
                description="Average degree of consolidation",
                unit="-",
                is_primary=True,
                expected_range=(0.0, 1.0),
            ),
            OutputVariable(
                name="u_mid",
                symbol="u_e(H/2)",
                description="Excess pore pressure at mid-depth",
                unit="kPa",
                is_primary=False,
                expected_range=(0.0, 200.0),
            ),
        ],
        governing_equation=(
            "Terzaghi 1-D consolidation PDE:\n"
            "  du_e/dt = c_v * d^2 u_e / dz^2\n"
            "  c_v = k / (m_v * gamma_w)\n"
            "  BC: u_e(0,t) = 0 (drained top), du_e/dz(H,t) = 0 (impermeable base)\n"
            "  IC: u_e(z,0) = delta_sigma\n"
            "  U(t) = 1 - sum_{m=0}^inf (2/M^2) exp(-M^2 T_v)\n"
            "  S(t) = U(t) * m_v * delta_sigma * H_dr"
        ),
        pde_type="parabolic",
        spatial_dim=1,
        is_steady=False,
        # The consolidation PDE is linear in excess pore pressure for fixed
        # coefficients, but the benchmark response is nonlinear in the inputs
        # through c_v=k/(m_v*gamma_w), T_v=c_v*t/H_dr^2, and S=U*m_v*Delta*sigma*H.
        is_linear=False,
        has_nonlinear_bc=False,
        physics_references=[
            "Terzaghi, K. (1943). Theoretical Soil Mechanics. Wiley. Chapter 12.",
            "Das, B. M. (2010). Principles of Geotechnical Engineering, 7th ed. "
            "Cengage. Chapter 11, Section 11.6.",
            "Craig, R. F. (2004). Craig's Soil Mechanics, 7th ed. Spon Press. "
            "Chapter 7, Section 7.2-7.4.",
            "Itasca FLAC3D one-dimensional consolidation verification example, "
            "used as Biot-theory, Darcy-flow, drained-boundary, impervious-base, "
            "and numerical-vs-analytical pore-pressure comparison boundary "
            "context, not calibration data for this benchmark surrogate. "
            "https://docs.itascacg.com/itasca940/flac3d/zone/test3d/Fluid/1DConsolidation/1dconsolidation.html",
            "MOLE Terzaghi one-dimensional consolidation analytical benchmark. "
            "https://mole-docs.readthedocs.io/en/main/examples/Parabolic/1D/Terzaghi1D.html",
            "OpenGTI consolidation overview for single-vs-double-drainage choice, "
            "drainage-path, secondary-compression, anisotropy, staged-loading, "
            "and field/lab parameter-boundary context "
            "(not validation data for this benchmark surrogate). "
            "https://opengti.io/consolidation",
            "OpenGeoSys hydro-mechanics benchmarks and ConsolidationBenchmark project "
            "documentation, including the published 'Consolidation benchmark with "
            "the staggered scheme', for staggered/monolithic coupled "
            "pressure-displacement HM/Biot "
            "omitted-scope context (not validation data for this one-dimensional "
            "benchmark); the OGS project file explicitly uses HYDRO_MECHANICS, "
            "staggered pressure/displacement coupling, fixed-stress stabilization, "
            "Backward-Euler time stepping, and displacement/stress/strain/velocity "
            "outputs that this analytical Terzaghi control does not solve. "
            "https://www.opengeosys.org/docs/benchmarks/hydro-mechanics/; "
            "https://www.opengeosys.org/stable/docs/benchmarks/hydro-mechanics/; "
            "https://doxygen.opengeosys.org/6.5.6/d7/df1/ogs_ctest_prj__HydroMechanics__StaggeredScheme__ConsolidationBenchmark__consolidation_benchmark__prj.html",
        ],
        solver_library="numpy (Terzaghi analytical Fourier series)",
        solver_details="Direct Terzaghi single-drainage Fourier-series evaluation "
        "with adaptive term count; n_z/n_t fidelity arguments are accepted for "
        "backward-compatible API shape but do not change the analytical response; "
        "excludes single-vs-double-drainage design selection, radial drains, "
        "layered/anisotropic soils, stress-dependent k/m_v, finite-strain or "
        "secondary compression, construction staging, 2-D/3-D coupled Biot "
        "poromechanics, staggered/monolithic pressure-displacement HM solves, "
        "fixed-stress stabilization, Backward-Euler time stepping, displacement/"
        "stress/strain/velocity field outputs, stress/strain field recovery, unsaturated flow, field-instrument "
        "interpretation, and laboratory/field oedometer parameter fitting; named "
        "_validate_physical_inputs private "
        "physical-input evaluation rejects malformed arrays and "
        "k/m_v/H_dr/delta_s/t_obs values outside the declared benchmark domain "
        "before Terzaghi Fourier-series evaluation; non-finite, non-physical, "
        "or out-of-contract S_t/U_t/u_mid responses are rejected before dispatch "
        "can archive a row",
        fidelity_levels=["default", "high"],
        typical_eval_time_ms={"default": 1.0, "high": 1.0},
        cost_level="L0",
        supports_batch=True,
        supports_gradient=False,
        supports_multifidelity=True,
        response_topology=["smooth_monotone", "strong_interaction", "heterogeneous"],
        effective_dim=5,
        important_variables=["m_v", "delta_s", "k", "H_dr", "t_obs"],
        interaction_pairs=[
            ("k", "m_v"),
            ("k", "H_dr"),
            ("k", "t_obs"),
            ("m_v", "H_dr"),
            ("m_v", "t_obs"),
            ("m_v", "delta_s"),
            ("H_dr", "t_obs"),
            ("H_dr", "delta_s"),
            ("delta_s", "k"),
            ("delta_s", "t_obs"),
        ],
        has_failure_boundary=True,
        failure_condition="S_t > 100 mm (exceeds allowable settlement for many foundation types)",
        sobol_S1={"k": 0.075, "m_v": 0.341, "H_dr": 0.058, "delta_s": 0.109, "t_obs": 0.005},
        sobol_ST={"k": 0.251, "m_v": 0.612, "H_dr": 0.170, "delta_s": 0.266, "t_obs": 0.061},
        sobol_computation="salib_saltelli_n1024_seed42_S_t_output",
        applicable_scenarios=["VS-A", "VS-C", "VS-G"],
        scenario_notes={
            "VS-A": "U(T_v) is steep near T_v ~ 0.2 and flat beyond T_v ~ 2; "
            "this creates strong heterogeneity in the response surface "
            "— space-filling designs must cover the transition zone.",
            "VS-C": "Nonlinear coupling between k, m_v, H_dr, t_obs through "
            "T_v = k*t/(m_v*gamma_w*H_dr^2) makes surrogate fitting challenging.",
            "VS-G": "Settlement limit S_t > 100 mm defines an implicit failure surface "
            "in 5D; this scenario evaluates whether sequential reliability "
            "methods estimate P_f efficiently under the preregistered protocol.",
        },
        recommended_n_train={"default": 50},
        n_saturation=200,
        noise_factor_candidates=["delta_s"],
        control_factor_candidates=["k", "m_v", "H_dr", "t_obs"],
        reference_solution_available=True,
        reference_grid_size=5000,
    )

    def _validate_physical_inputs(self, X_phys: np.ndarray) -> np.ndarray:
        X_phys = np.asarray(X_phys, dtype=float)
        if X_phys.ndim != 2 or X_phys.shape[1] != self.meta.dim_input:
            raise ValueError(
                "soil_consolidation_5d expects a physical input array with shape "
                f"(n, {self.meta.dim_input}); got {X_phys.shape!r}"
            )

        tol = 1e-10
        k = X_phys[:, 0]
        m_v = X_phys[:, 1]
        H_dr = X_phys[:, 2]
        delta_s = X_phys[:, 3]
        t_obs = X_phys[:, 4]
        if (
            not np.all(np.isfinite(X_phys))
            or np.any(k < 1e-10 * (1.0 - tol))
            or np.any(k > 1e-7 * (1.0 + tol))
            or np.any(m_v < 1e-7 * (1.0 - tol))
            or np.any(m_v > 5e-6 * (1.0 + tol))
            or np.any(H_dr < 2.0 - tol)
            or np.any(H_dr > 20.0 + tol)
            or np.any(delta_s < 20.0 - tol)
            or np.any(delta_s > 200.0 + tol)
            or np.any(t_obs < 0.5 - tol)
            or np.any(t_obs > 50.0 + tol)
        ):
            raise ValueError(
                "soil_consolidation_5d physical inputs must stay inside the declared "
                "k/m_v/H_dr/delta_s/t_obs benchmark domain before Terzaghi evaluation"
            )
        return X_phys

    def _evaluate_core(self, X_phys: np.ndarray, fidelity: str = "default", **kwargs) -> np.ndarray:
        X_phys = self._validate_physical_inputs(X_phys)

        if fidelity == "high":
            n_z, n_t = 400, 12000
        else:
            n_z, n_t = 100, 3000
        n = X_phys.shape[0]
        Y = np.full((n, 3), np.nan)
        for i in range(n):
            try:
                response = self._solve_consolidation(
                    k=float(X_phys[i, 0]),
                    m_v=float(X_phys[i, 1]),
                    H_dr=float(X_phys[i, 2]),
                    delta_s=float(X_phys[i, 3]),
                    t_obs=float(X_phys[i, 4]),
                    n_z=n_z,
                    n_t=n_t,
                )
                if response.shape != (self.meta.dim_output,) or not np.all(np.isfinite(response)):
                    raise RuntimeError("non-finite consolidation response")
                delta_s_kpa = float(X_phys[i, 3])
                S_final_mm = float(X_phys[i, 1]) * delta_s_kpa * 1e3 * float(X_phys[i, 2]) * 1e3
                if (
                    response[0] < 0.0
                    or response[0] > S_final_mm * (1.0 + 1e-12)
                    or response[1] < 0.0
                    or response[1] > 1.0
                    or response[2] < 0.0
                    or response[2] > delta_s_kpa * (1.0 + 1e-12)
                ):
                    raise RuntimeError(
                        "non-physical consolidation response "
                        f"S_t/U_t/u_mid={np.array2string(response, precision=6, separator=', ')}"
                    )
                output_ranges = np.asarray(
                    [output.expected_range for output in self.meta.output_variables],
                    dtype=float,
                )
                below = response < (output_ranges[:, 0] - 1e-10)
                above = response > (output_ranges[:, 1] + 1e-10)
                if np.any(below) or np.any(above):
                    raise RuntimeError(
                        "out-of-contract consolidation S_t/U_t/u_mid response "
                        "outside the declared output ranges"
                    )
                Y[i] = response
            except Exception:
                x_text = np.array2string(X_phys[i], precision=6, separator=", ")
                raise RuntimeError(
                    "soil_consolidation_5d Terzaghi analytical consolidation solve "
                    f"failed for physical input {x_text}; refusing silent NaN "
                    "output because the formal benchmark response surface "
                    "requires finite settlement and pore-pressure values."
                ) from None
        return Y

    @staticmethod
    def _solve_consolidation(
        k: float,
        m_v: float,
        H_dr: float,
        delta_s: float,
        t_obs: float,
        n_z: int = 100,
        n_t: int = 3000,
    ) -> np.ndarray:
        """Evaluate the Terzaghi single-drainage analytical solution.

        Parameters
        ----------
        k : float
            Hydraulic conductivity (m/s).
        m_v : float
            Volume compressibility (1/Pa).
        H_dr : float
            Drainage path length (m).
        delta_s : float
            Surcharge load (kPa).
        t_obs : float
            Observation time (years).
        n_z : int
            Backward-compatible fidelity argument; analytical output is grid independent.
        n_t : int
            Backward-compatible fidelity argument; analytical output is time-step independent.

        Returns
        -------
        result : ndarray, shape (3,)
            [S_t (mm), U_t (-), u_mid (kPa)]
        """
        if (
            not np.all(np.isfinite([k, m_v, H_dr, delta_s, t_obs]))
            or k <= 0.0
            or m_v <= 0.0
            or H_dr <= 0.0
            or delta_s < 0.0
            or t_obs < 0.0
            or n_z <= 0
            or n_t <= 0
        ):
            raise ValueError("invalid Terzaghi consolidation physical input")

        # Coefficient of consolidation (m²/s)
        c_v = k / (m_v * _GAMMA_W)
        t_total = t_obs * 365.25 * 24.0 * 3600.0
        delta_s_Pa = delta_s * 1e3
        T_v = c_v * t_total / (H_dr**2)

        if delta_s_Pa == 0.0:
            return np.array([0.0, 0.0, 0.0])

        # Use enough odd Fourier terms to resolve small time factors while
        # naturally truncating large-time-factor exponentials to zero.
        n_terms = 2000
        m = np.arange(n_terms, dtype=float)
        M = np.pi * (2.0 * m + 1.0) / 2.0
        exp_terms = np.exp(-(M**2) * T_v)
        U_t = 1.0 - float(np.sum(2.0 / M**2 * exp_terms))
        if U_t < -1e-10 or U_t > 1.0 + 1e-10:
            raise ValueError(f"non-physical Terzaghi degree of consolidation U={U_t!r}")
        U_t = min(max(U_t, 0.0), 1.0)

        # Settlement (mm)
        S_final = m_v * delta_s_Pa * H_dr * 1e3  # mm
        S_t = U_t * S_final

        z_mid = H_dr / 2.0
        u_mid_Pa = float(np.sum((2.0 * delta_s_Pa / M) * np.sin(M * z_mid / H_dr) * exp_terms))
        if u_mid_Pa < -1e-6 or u_mid_Pa > delta_s_Pa + 1e-6:
            raise ValueError(f"non-physical Terzaghi mid-depth pore pressure u_mid={u_mid_Pa!r}")
        u_mid = min(max(u_mid_Pa, 0.0), delta_s_Pa) * 1e-3  # Pa → kPa

        return np.array([S_t, U_t, u_mid])
