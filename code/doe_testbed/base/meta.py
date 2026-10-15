"""
doe_testbed.base.meta
=======================
Data-class definitions for case metadata.

Contains:
  InputVariable  — Full description of a single input variable (units, bounds, distribution, DOE role)
  OutputVariable — Full description of a single output variable (units, failure threshold)
  CaseMeta       — Case-level metadata (physics model, cost, response topology, VS scenarios)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# InputVariable
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class InputVariable:
    """Full physical description of a single input variable.

    Attributes
    ----------
    name : str
        Variable name (valid Python identifier, snake_case).
    symbol : str
        Mathematical symbol (LaTeX) for plots and docs.
    description : str
        Physical meaning (one sentence).
    unit : str
        SI unit; use "-" for dimensionless quantities.
    lower : float
        Physical lower bound.
    upper : float
        Physical upper bound.
    log_scale : bool
        If True, normalise in log-space (e.g. Reynolds number spanning decades).
    distribution : str
        Suggested prior distribution: 'uniform' | 'loguniform' | 'normal'.
    variable_type : str
        Variable type: 'continuous' | 'integer' | 'categorical'.
    role : str
        Taguchi role: 'design' | 'noise' | 'both'.
    """

    name: str
    symbol: str
    description: str
    unit: str
    lower: float
    upper: float
    log_scale: bool = False
    distribution: str = "uniform"
    variable_type: str = "continuous"
    role: str = "design"

    def normalize(self, x_phys: float) -> float:
        """Physical value → [0, 1] normalised space."""
        if self.log_scale:
            lo = math.log10(self.lower)
            hi = math.log10(self.upper)
            return (math.log10(x_phys) - lo) / (hi - lo)
        return (x_phys - self.lower) / (self.upper - self.lower)

    def denormalize(self, u: float) -> float:
        """[0, 1] normalised space → physical value."""
        if self.log_scale:
            lo = math.log10(self.lower)
            hi = math.log10(self.upper)
            return 10.0 ** (lo + u * (hi - lo))
        return self.lower + u * (self.upper - self.lower)


# ─────────────────────────────────────────────────────────────────────────────
# OutputVariable
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class OutputVariable:
    """Full description of a single output variable.

    Attributes
    ----------
    name : str
        Output variable name (valid Python identifier).
    symbol : str
        Mathematical symbol (LaTeX).
    description : str
        Physical meaning (one sentence).
    unit : str
        SI unit; use "-" for dimensionless quantities.
    is_primary : bool
        Whether this is the primary output of interest.
    expected_range : tuple, optional
        Physically plausible output range (lo, hi) for outlier detection; None to skip.
    failure_threshold : float, optional
        Failure threshold; None means no failure criterion.
    failure_direction : str, optional
        Failure direction: 'above' (exceeds threshold) | 'below' (falls below) | None.
    monotone_direction : str, optional
        Known monotone direction w.r.t. all important inputs:
        'increase' | 'decrease' | None. Used for VS-B effect estimation ground truth.
    """

    name: str
    symbol: str
    description: str
    unit: str
    is_primary: bool = True
    expected_range: Optional[Tuple[float, float]] = None
    failure_threshold: Optional[float] = None
    failure_direction: Optional[str] = None
    monotone_direction: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# CaseMeta
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CaseMeta:
    """Case-level metadata -- the central knowledge record for each test problem.

    Every case must define a complete CaseMeta.  It serves as:
      * The basis for auto-registration and discoverability
      * The filter target for list_cases() queries
      * The entry point for DOE researchers to understand case characteristics
      * The container for Sobol index caches and verification records

    Fields are organised into 7 groups: identity, dimensions, physics model,
    computational cost, response-surface traits, VS scenario applicability,
    and reference-solution information.
    """

    # ── Identity ───────────────────────────────────────────────────────────
    case_id: str
    """Globally unique identifier, snake_case, e.g. 'airfoil_stall_2d'."""

    name: str
    """Human-readable name."""

    version: str = "1.0.0"
    """Case version (semantic versioning)."""

    domain: str = "unknown"
    """Discipline domain.
    Enum: aerodynamics | structures | heat_transfer | chemical_process |
          geomechanics | multidisciplinary
    (Reserved: combustion | acoustics | electromagnetics)
    """

    tags: List[str] = field(default_factory=list)
    """Free-form tag list for search/filter, e.g. ['airfoil', 'stall', 'threshold']."""

    # ── Dimensions ─────────────────────────────────────────────────────────
    dim_input: int = 0
    """Number of input dimensions."""

    dim_output: int = 1
    """Number of output dimensions."""

    input_variables: List[InputVariable] = field(default_factory=list)
    """Ordered list of input variable descriptors (column order matches evaluate())."""

    output_variables: List[OutputVariable] = field(default_factory=list)
    """Ordered list of output variable descriptors (column order matches evaluate() return)."""

    # ── Physics model ─────────────────────────────────────────────────────
    governing_equation: str = ""
    """Governing equation name, e.g. 'Euler-Bernoulli beam theory'."""

    pde_type: str = "unknown"
    """PDE type: elliptic | parabolic | hyperbolic | ode_system |
               algebraic | integral | hybrid | analytical
    """

    spatial_dim: int = 1
    """Physical spatial dimension (1D/2D/3D), distinct from input dimension."""

    is_steady: bool = True
    """Whether the problem is steady-state."""

    is_linear: bool = False
    """Whether the governing equation is linear."""

    has_nonlinear_bc: bool = False
    """Whether nonlinear boundary conditions are present."""

    physics_references: List[str] = field(default_factory=list)
    """Literature references for the physics model, format: ['Author, Year, Journal, DOI']."""

    solver_library: str = ""
    """Underlying solver library, e.g. 'neuralfoil', 'scipy.integrate', 'scikit-fem'."""

    solver_details: str = ""
    """Solver configuration (mesh size, accuracy settings, etc.)."""

    fidelity_levels: List[str] = field(default_factory=lambda: ["default"])
    """Supported fidelity levels, e.g. ['low', 'high']. Single-fidelity cases use ['default']."""

    # ── Computational cost ─────────────────────────────────────────────────
    typical_eval_time_ms: Dict[str, float] = field(default_factory=lambda: {"default": 1.0})
    """Typical single-evaluation time (ms) per fidelity level.
    Measured on single-core CPU, Intel i7-12700, Python 3.11."""

    cost_level: str = "L0"
    """Cost level: L0 (analytic/lightweight) | L1 (numerical PDE/ODE) | L2 (50-500 ms) | L3 (0.5-10 s)."""

    supports_batch: bool = True
    """Whether evaluate() accepts batch N*d inputs and returns N*m outputs.
    This flag indicates whether the internal solver is vectorised."""

    supports_gradient: bool = False
    """Whether analytic gradients are available (via evaluate_gradient())."""

    supports_multifidelity: bool = False
    """Whether a fidelity parameter enables multi-fidelity evaluation."""

    # ── Response-surface traits ─────────────────────────────────────────────
    response_topology: List[str] = field(default_factory=list)
    """Response-surface topology traits (a case may have multiple).
    Enum: smooth_monotone | smooth_unimodal | threshold_jump |
          discontinuous | strong_interaction | sparse_effects |
          constrained_domain | multi_modal | heterogeneous
    """

    effective_dim: int = 0
    """Effective dimension: number of variables truly affecting the output (0 = dim_input)."""

    important_variables: List[str] = field(default_factory=list)
    """Names of important variables (Sobol total index ST > 0.05)."""

    interaction_pairs: List[Tuple[str, str]] = field(default_factory=list)
    """Variable pairs with significant interaction effects, e.g. [('alpha', 'Re')]."""

    has_failure_boundary: bool = False
    """Whether an explicit failure boundary is defined."""

    failure_condition: Optional[str] = None
    """Failure condition description, e.g. 'C_L < 0.5 (lift drops after stall)'."""

    # ── Sobol indices ───────────────────────────────────────────────────────
    sobol_S1: Optional[Dict[str, float]] = None
    """First-order Sobol indices (uniform input); None if not yet computed."""

    sobol_ST: Optional[Dict[str, float]] = None
    """Total Sobol indices (including higher-order interactions); None if not yet computed."""

    sobol_computation: str = "salib_saltelli_n10000"
    """Description of the Sobol index computation method."""

    # ── VS scenario applicability ───────────────────────────────────────────
    applicable_scenarios: List[str] = field(default_factory=list)
    """DOE validation scenarios applicable to this case.
    Enum: VS-A | VS-B | VS-C | VS-D | VS-E | VS-F | VS-G | VS-H
    """

    scenario_notes: Dict[str, str] = field(default_factory=dict)
    """Per-scenario notes, e.g. {'VS-G': 'Stall boundary is the key sequential-method target'}."""

    recommended_n_train: Dict[str, int] = field(default_factory=dict)
    """Recommended training set size per fidelity (for acceptable surrogate accuracy)."""

    n_saturation: int = 200
    """Saturation sample count: beyond this, surrogate error plateaus."""

    noise_factor_candidates: List[str] = field(default_factory=list)
    """Variable names suitable as Taguchi noise factors."""

    control_factor_candidates: List[str] = field(default_factory=list)
    """Variable names suitable as Taguchi control factors."""

    # ── Reference solution ─────────────────────────────────────────────────
    reference_solution_available: bool = True
    """Whether a dense reference solution can be generated (for accurate RMSE computation)."""

    reference_grid_size: int = 10000
    """Recommended grid point count for the reference solution."""

    scalable_dimension: bool = False
    """Whether the problem supports dimension scaling (e.g. 5D/10D/20D from the same physics)."""

    scalable_dimension_range: Optional[Tuple[int, int]] = None
    """Scalable dimension range (min_d, max_d), e.g. (3, 50)."""

    # ── Post-init validation ─────────────────────────────────────────────
    def __post_init__(self) -> None:
        """Lightweight consistency checks (runs automatically after __init__).

        Validates:
          1. applicable_scenarios are all in VS_ALL
          2. cost_level in {L0, L1, L2, L3}
          3. If scalable_dimension=True, scalable_dimension_range must be set
        """
        # ── 1. VS scenario validity ──────────────────────────────────────────
        if self.applicable_scenarios:
            from doe_testbed.base.scenarios import validate_scenarios

            # In-place normalisation (deduplicate, preserve order)
            self.applicable_scenarios = validate_scenarios(self.applicable_scenarios)

        # ── 2. Cost level validity ───────────────────────────────────────────
        if self.cost_level not in ("L0", "L1", "L2", "L3"):
            raise ValueError(
                f"[{self.case_id}] cost_level='{self.cost_level}' is invalid; "
                f"valid values: L0 / L1 / L2 / L3"
            )

        # ── 3. Scalable dimension consistency ────────────────────────────────
        if self.scalable_dimension and self.scalable_dimension_range is None:
            raise ValueError(
                f"[{self.case_id}] scalable_dimension=True but "
                f"scalable_dimension_range=(min_d, max_d) is not set"
            )

    # ── Convenience properties ─────────────────────────────────────────────

    @property
    def input_names(self) -> List[str]:
        """List of all input variable names."""
        return [v.name for v in self.input_variables]

    @property
    def output_names(self) -> List[str]:
        """List of all output variable names."""
        return [v.name for v in self.output_variables]

    @property
    def bounds_lower(self):
        """Physical lower bounds array, shape (dim_input,)."""
        import numpy as np

        return np.array([v.lower for v in self.input_variables])

    @property
    def bounds_upper(self):
        """Physical upper bounds array, shape (dim_input,)."""
        import numpy as np

        return np.array([v.upper for v in self.input_variables])
