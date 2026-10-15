"""
doe_testbed.registry.autodiscover
=====================================
Auto-scan all Python modules under cases/ and import them,
triggering @register_case decorators to complete registration.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path


def discover() -> None:
    """Walk all sub-packages/modules under doe_testbed/cases/ and
    import each one to trigger @register_case registration.
    """
    import doe_testbed.cases as cases_pkg

    cases_root = Path(cases_pkg.__file__).parent

    for _, module_name, is_pkg in pkgutil.walk_packages(
        path=[str(cases_root)],
        prefix="doe_testbed.cases.",
        onerror=lambda name: None,
    ):
        if not is_pkg:
            try:
                importlib.import_module(module_name)
            except Exception as exc:
                # A single case import failure should not break the whole library
                import warnings

                warnings.warn(
                    f"[autodiscover] Error importing '{module_name}': {exc}",
                    ImportWarning,
                    stacklevel=2,
                )
