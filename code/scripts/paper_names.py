# -*- coding: utf-8 -*-
"""Canonical human-readable display names for the manuscript.

Single source of truth for the publication names shown in the main text,
the Supplementary Information tables and the figures, so that every
display item uses identical wording.  Raw repository identifiers remain
in the generated ``paper/si_*.md`` mirrors and in the archives, so the
mapping stays fully traceable.
"""
from __future__ import annotations

SAMPLER_LABELS = {
    "random": "Random",
    "lhs": "LHS",
    "lhs_opt": "Optimised LHS",
    "sobol": "Sobol'",
    "halton": "Halton",
}

# Short surrogate labels used inside composed method names ("LHS + GP").
SURROGATE_LABELS = {
    "gp": "GP",
    "gp_rbf": "GP (RBF kernel)",
    "rbf": "RBF",
    "svr": "SVR",
    "rf": "Random forest",
    "gb": "Gradient boosting",
    "knn": "kNN",
    "mlp": "MLP",
    "linear": "Linear",
    "poly2": "Polynomial (deg. 2)",
    "poly3": "Polynomial (deg. 3)",
    "poly5": "Polynomial (deg. 5)",
}

# Full surrogate/kernel names for the Supplementary Table S1 column.
SURROGATE_FULL_LABELS = {
    "gp": "Gaussian process",
    "gp_rbf": "Gaussian process (RBF kernel)",
    "botorch_singletaskgp": "Gaussian process (BoTorch SingleTaskGP)",
    "botorch_saas": "Gaussian process (SAAS prior, BoTorch)",
    "rbf": "RBF interpolation",
    "svr": "Support vector regression",
    "rf": "Random forest",
    "gb": "Gradient boosting",
    "knn": "k-nearest neighbours",
    "mlp": "Multilayer perceptron",
    "linear": "Linear regression",
    "poly": "Polynomial (ridge)",
    "poly_chaos": "Polynomial chaos",
}

SPECIAL_METHOD_LABELS = {
    "max_variance": "Max-variance AL",
    "akmcs": "AK-MCS",
    "ego_ei": "EGO (EI)",
    "botorch_qei": "BoTorch qEI",
    "botorch_saasbo": "SAASBO",
}

# Method-pool family display names. Families whose identifier contains an
# abbreviation must not fall through to the generic title-case fallback,
# which would render sequential_ego as Sequential Ego.
FAMILY_LABELS = {
    "static_design": "Static Design",
    "sequential_ego": "Sequential EGO",
    "sequential_botorch_ei": "Sequential BoTorch EI",
    "sequential_botorch_saas": "Sequential BoTorch SAAS",
    "sequential_active_learning": "Sequential Active Learning",
    "sequential_reliability": "Sequential Reliability",
}

CASE_LABELS = {
    "airfoil_stall_2d": "Airfoil stall polar",
    "waverider_4d": "Waverider forebody",
    "airfoil_transonic_5d": "Transonic airfoil drag",
    "supersonic_intake_5d": "Supersonic intake",
    "rotor_bemt_6d": "Rotor blade-element momentum",
    "wing_vlm_6d": "Finite-wing lifting line",
    "rocket_isp_4d": "Rocket specific impulse",
    "laminar_flame_speed_5d": "Laminar flame speed",
    "membrane_reactor_5d": "Membrane reactor",
    "supersonic_nozzle_5d": "Supersonic nozzle",
    "fixed_bed_reactor_6d": "Fixed-bed reactor",
    "soil_consolidation_5d": "Soil consolidation",
    "dam_seepage_6d": "Dam seepage",
    "groundwater_contamination_6d": "Groundwater contamination",
    "oil_reservoir_9d": "Oil-reservoir seepage",
    "fin_heat_sink_4d": "Fin heat sink",
    "multilayer_wall_4d": "Multilayer wall (4-layer)",
    "multilayer_wall_8d": "Multilayer wall (8-layer)",
    "multilayer_wall_16d": "Multilayer wall (16-layer)",
    "casting_solidification_5d": "Casting solidification",
    "hetero_conduction_2d": "Heterogeneous conduction",
    "battery_thermal_7d": "Battery thermal management",
    "pressure_vessel_4d": "Pressure vessel",
    "solar_pv_yield_6d": "Solar photovoltaic yield",
    "wind_turbine_power_6d": "Wind-turbine power",
    "aircraft_conceptual_8d": "Aircraft conceptual design",
    "tuned_mass_damper_3d": "Tuned mass damper",
    "cantilever_beam_5d": "Cantilever beam",
    "plate_with_hole_5d": "Plate with hole",
    "thin_plate_buckling_5d": "Thin-plate buckling",
    "variable_beam_5d": "Variable-section beam (5-D)",
    "variable_beam_10d": "Variable-section beam (10-D)",
    "variable_beam_20d": "Variable-section beam (20-D)",
    "ten_bar_truss_10d": "Ten-bar truss",
    # Analytical anchor functions (D1 transfer analysis only).
    "branin_2d": "Branin (d=2)",
    "ishigami_3d": "Ishigami (d=3)",
    "friedman_5d": "Friedman (d=5)",
    "hartmann_6d": "Hartmann (d=6)",
    "sobol_g_8d": "Sobol' g-function (d=8)",
    "wing_weight_10d": "Wing weight (d=10)",
}

DOMAIN_LABELS = {
    "aerodynamics": "Aerodynamics",
    "chemical_process": "Chemical process",
    "geomechanics": "Geomechanics",
    "heat_transfer": "Heat transfer",
    "multidisciplinary": "Multidisciplinary",
    "structures": "Structures",
}

EVIDENCE_TIER_LABELS = {
    "external_note_analytic_control": "External note + analytic control",
    "external_note_targeted_regression": "External note + targeted regression",
    "targeted_analytical_control": "Targeted analytical control",
    "targeted_plausibility_only": "Targeted plausibility only",
    "not_listed": "Not listed",
}


def method_label(method: str) -> str:
    """Display name for a method id (e.g. ``lhs_opt+rbf`` -> ``Optimised LHS + RBF``)."""
    m = str(method).strip()
    key = m.lower()
    if key in SPECIAL_METHOD_LABELS:
        return SPECIAL_METHOD_LABELS[key]
    if "+" in key:
        sampler, surrogate = key.split("+", 1)
        s_lab = SAMPLER_LABELS.get(sampler, sampler.replace("_", " ").title())
        g_lab = SURROGATE_LABELS.get(surrogate, surrogate.replace("_", " ").upper())
        return f"{s_lab} + {g_lab}"
    return m.replace("_", " ")


def family_label(family: str) -> str:
    """Display name for a method-pool family id (a sampler or a sequential strategy)."""
    key = str(family).strip().lower()
    if key in SPECIAL_METHOD_LABELS:
        return SPECIAL_METHOD_LABELS[key]
    if key in FAMILY_LABELS:
        return FAMILY_LABELS[key]
    return SAMPLER_LABELS.get(key, key.replace("_", " ").title())


def sampler_label(sampler: str) -> str:
    key = str(sampler).strip().lower()
    return SAMPLER_LABELS.get(key, key.replace("_", " ").title())


def surrogate_label(surrogate: str) -> str:
    key = str(surrogate).strip().lower()
    return SURROGATE_FULL_LABELS.get(key, key.replace("_", " "))


def case_label(case_id: str) -> str:
    key = str(case_id).strip()
    return CASE_LABELS.get(key, key.replace("_", " "))


def domain_label(domain: str) -> str:
    key = str(domain).strip().lower()
    return DOMAIN_LABELS.get(key, key.replace("_", " ").capitalize())


def tier_label(tier: str) -> str:
    key = str(tier).strip()
    return EVIDENCE_TIER_LABELS.get(key, key.replace("_", " "))


def topology_label(topology: str) -> str:
    """One or more comma-separated topology tags -> readable text."""
    return ", ".join(
        t.strip().replace("_", " ") for t in str(topology).split(",") if t.strip()
    )
