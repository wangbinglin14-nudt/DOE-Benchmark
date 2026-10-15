"""Case registry: registration decorator plus list_cases()/get_case() lookup."""
from doe_testbed.registry import autodiscover
from doe_testbed.registry.registry import (
    describe_case,
    get_case,
    get_minimum_covering_set,
    list_cases,
    print_cases_table,
    register_case,
)

__all__ = [
    "register_case",
    "get_case",
    "list_cases",
    "describe_case",
    "get_minimum_covering_set",
    "print_cases_table",
    "autodiscover",
]
