"""
doe_testbed — Engineering test problem library for DOE method evaluation
=======================================================================

A curated collection of 34 physics-grounded engineering test problems
under a unified interface.  This library provides *test problems only*;
DOE methods (LHS, Sobol sequences, Bayesian optimisation, surrogate
modelling, sensitivity analysis, etc.) should come from external
packages such as ``scipy.stats.qmc``, ``SALib``, or ``scikit-learn``.

Users generate design points in the normalised ``[0, 1]^d`` space using
their own DOE method, then call ``case.evaluate(X)`` to obtain physical
simulation responses.

Quick start
-----------
>>> import doe_testbed as db
>>> db.list_cases()                              # all registered cases
>>> case = db.get_case("cantilever_beam_5d")
>>> X = case.sample(50, method="lhs", seed=42)   # normalised space, (50, 5)
>>> Y = case.evaluate(X)                          # physical-unit outputs, (50, 3)
>>> case.describe()                               # print case summary
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("doe-testbed")
except PackageNotFoundError:
    __version__ = "0.4.0.dev0"

from doe_testbed.base.case import BaseCase

# ── Data classes (for user type annotations) ────────────────────────────────
from doe_testbed.base.meta import CaseMeta, InputVariable, OutputVariable

# ── Trigger auto-registration (import all case sub-packages) ────────────────
from doe_testbed.registry import autodiscover

# ── Core public API ──────────────────────────────────────────────────────────
from doe_testbed.registry.registry import (
    describe_case,
    get_case,
    get_case_meta,
    get_minimum_covering_set,
    list_cases,
    print_cases_table,
)

autodiscover.discover()

__all__ = [
    "__version__",
    "get_case",
    "get_case_meta",
    "list_cases",
    "describe_case",
    "get_minimum_covering_set",
    "print_cases_table",
    "CaseMeta",
    "InputVariable",
    "OutputVariable",
    "BaseCase",
]
