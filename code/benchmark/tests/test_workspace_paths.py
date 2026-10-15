"""Paper artifact paths stay portable after workspace folder organization."""
from pathlib import Path
import pytest
from scripts import prepare_figdata, fill_paper_numbers

@pytest.mark.parametrize("module", [prepare_figdata, fill_paper_numbers])
def test_logical_and_resolved_paper_paths_are_portable(module):
    path = module.ROOT / "paper" / "numbers.tex"
    assert module._portable_path(path) == "paper/numbers.tex"
    assert module._portable_path(path.resolve()) == "paper/numbers.tex"
