"""Base interfaces: BaseCase, CaseMeta and input/output variable schemas."""
from doe_testbed.base.case import BaseCase
from doe_testbed.base.meta import CaseMeta, InputVariable, OutputVariable
from doe_testbed.base.scenarios import (
    VS_A,
    VS_ALL,
    VS_B,
    VS_C,
    VS_D,
    VS_DESCRIPTIONS,
    VS_E,
    VS_F,
    VS_G,
    VS_H,
    validate_scenarios,
)

__all__ = [
    "CaseMeta",
    "InputVariable",
    "OutputVariable",
    "BaseCase",
    "VS_A",
    "VS_B",
    "VS_C",
    "VS_D",
    "VS_E",
    "VS_F",
    "VS_G",
    "VS_H",
    "VS_ALL",
    "VS_DESCRIPTIONS",
    "validate_scenarios",
]
