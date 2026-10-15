"""The archived method-family classification must survive manifest reproduction."""
import pandas as pd
import pytest
from scripts.prepare_figdata import make_fig1


def test_manifest_catalog_preserves_archived_method_family(tmp_path):
    df = pd.DataFrame({"method": ["lhs+gp", "lhs+gp", "ego_ei"],
                       "method_family": ["static_design", "static_design", "sequential_ego"]})
    make_fig1(df, tmp_path, declared_methods=["lhs+gp", "ego_ei"], method_source="test manifest")
    catalog = pd.read_csv(tmp_path / "fig1_method_catalog.csv").set_index("method")
    assert catalog.loc["lhs+gp", "family"] == "static_design"
    assert catalog.loc["ego_ei", "family"] == "sequential_ego"
    assert catalog.loc["lhs+gp", "surrogate"] == "gp"


def test_manifest_catalog_rejects_conflicting_archived_families(tmp_path):
    df = pd.DataFrame({"method": ["lhs+gp", "lhs+gp"],
                       "method_family": ["static_design", "sequential_ego"]})
    with pytest.raises(AssertionError, match="multiple families"):
        make_fig1(df, tmp_path, declared_methods=["lhs+gp"])
