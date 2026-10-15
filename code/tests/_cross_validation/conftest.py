"""Make pytest collect cross-validation modules named {case_id}.py
instead of the default test_*.py convention."""

import pytest


def pytest_collect_file(parent, file_path):
    """Collect any .py file in this directory as a test module."""
    if (
        file_path.suffix == ".py"
        and file_path.parent.name == "_cross_validation"
        and file_path.name not in ("__init__.py", "conftest.py")
    ):
        return pytest.Module.from_parent(parent, path=file_path)
