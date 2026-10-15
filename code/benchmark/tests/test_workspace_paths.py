"""Published manuscript paths must not depend on the caller's directory."""

import ast
import importlib
import os
from pathlib import Path

import pytest
from scripts import fill_paper_numbers, prepare_figdata, project_layout

CODE_ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPTS = CODE_ROOT.parent / "manuscripts"


def test_directory_configuration_uses_the_published_manuscripts():
    assert getattr(project_layout, "MANUSCRIPTS_DIR", None) == MANUSCRIPTS


@pytest.mark.parametrize(
    ("module_name", "attribute", "expected"),
    [
        ("prepare_figdata", "out_dir", MANUSCRIPTS / "figdata"),
        ("plots_nature", "figdata", MANUSCRIPTS / "figdata"),
        ("plots_nature", "out_dir", MANUSCRIPTS / "figures"),
        ("fill_paper_numbers", "figdata_dir", MANUSCRIPTS / "figdata"),
        ("fill_paper_numbers", "output_tex", MANUSCRIPTS / "numbers.tex"),
        ("fill_paper_numbers", "output_json", MANUSCRIPTS / "numbers.json"),
        ("generate_si", "output_dir", MANUSCRIPTS / "tables"),
    ],
)
def test_default_artifact_paths_are_independent_of_cwd(
    module_name, attribute, expected, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    module = importlib.import_module(f"scripts.{module_name}")
    assert getattr(module.parse_args([]), attribute) == expected


@pytest.mark.parametrize(
    ("module_name", "option", "attribute"),
    [
        ("prepare_figdata", "--out-dir", "out_dir"),
        ("plots_nature", "--figdata", "figdata"),
        ("plots_nature", "--out-dir", "out_dir"),
        ("fill_paper_numbers", "--figdata-dir", "figdata_dir"),
        ("fill_paper_numbers", "--output-tex", "output_tex"),
        ("fill_paper_numbers", "--output-json", "output_json"),
        ("generate_si", "--output-dir", "output_dir"),
    ],
)
def test_explicit_artifact_paths_remain_supported(module_name, option, attribute):
    module = importlib.import_module(f"scripts.{module_name}")
    assert getattr(module.parse_args([option, "custom-output"]), attribute) == Path("custom-output")


@pytest.mark.parametrize("module", [prepare_figdata, fill_paper_numbers])
def test_manuscript_paths_are_portable_relative_to_code(module):
    formatter = getattr(module, "portable_path", None)
    assert formatter is not None
    assert formatter(MANUSCRIPTS / "numbers.tex", CODE_ROOT) == ("../manuscripts/numbers.tex")


def test_archived_result_path_convention_is_unchanged():
    formatter = getattr(project_layout, "portable_path", None)
    assert formatter is not None
    path = CODE_ROOT / "benchmark" / "results" / "full_benchmark" / "results.csv"
    assert formatter(path, CODE_ROOT) == "benchmark/results/full_benchmark/results.csv"


def test_full_pipeline_uses_published_artifact_paths(tmp_path, monkeypatch):
    from scripts import run_pipeline

    monkeypatch.chdir(tmp_path)
    cfg = run_pipeline.resolve_config(run_pipeline.parse_args(["--profile", "full"]))
    assert (cfg.paper_dir, cfg.figdata_dir, cfg.figures_dir) == (
        MANUSCRIPTS,
        MANUSCRIPTS / "figdata",
        MANUSCRIPTS / "figures",
    )


def test_smoke_pipeline_does_not_write_to_published_manuscripts(tmp_path):
    from scripts import run_pipeline

    cfg = run_pipeline.resolve_config(
        run_pipeline.parse_args(["--profile", "smoke", "--sweep-dir", str(tmp_path)])
    )
    assert cfg.paper_dir.is_relative_to(tmp_path)
    assert not cfg.paper_dir.is_relative_to(MANUSCRIPTS)


def test_si_pipeline_command_and_expected_outputs_agree():
    from scripts import run_pipeline

    cfg = run_pipeline.resolve_config(run_pipeline.parse_args(["--profile", "full"]))
    stage = next(stage for stage in run_pipeline.build_stages(cfg) if stage.name == "si")
    command = stage.build_cmd(cfg)
    target = Path(command[command.index("--output-dir") + 1])
    if (target / "tables").is_dir():
        target = target / "tables"
    assert all(path.parent == target for path in stage.expected_outputs)


@pytest.mark.parametrize(
    ("script", "attribute", "expected"),
    [
        ("build_supplementary.py", "PAPER", MANUSCRIPTS),
        ("build_supplementary.py", "TABLES", MANUSCRIPTS / "tables"),
        ("generate_paper_figures.py", "OUT_DIR", MANUSCRIPTS / "figures" / "supplementary"),
        ("audit_macros.py", "PAPER", MANUSCRIPTS),
        ("audit_publication_readiness.py", "PAPER", MANUSCRIPTS),
        ("check_paper_draft_ready.py", "PAPER", MANUSCRIPTS),
        ("check_server_return_package.py", "PAPER_DIR_DEFAULT", MANUSCRIPTS),
        ("check_submission_ready.py", "PAPER_DIR_DEFAULT", MANUSCRIPTS),
        ("review_response_analyses.py", "PAPER", MANUSCRIPTS),
    ],
)
def test_generators_with_import_side_effects_use_published_paths(script, attribute, expected):
    """Evaluate only a path assignment, never execute a table/figure generator."""
    source = CODE_ROOT / "scripts" / script
    tree = ast.parse(source.read_text(encoding="utf-8-sig"))
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == attribute for target in node.targets)
    )
    values = {
        "MANUSCRIPTS_DIR": MANUSCRIPTS,
        "FIGURES_DIR": MANUSCRIPTS / "figures",
        "TABLES_DIR": MANUSCRIPTS / "tables",
        "ROOT": CODE_ROOT,
        "PAPER": CODE_ROOT / "paper",
        "__file__": str(source),
        "os": os,
    }
    expression = ast.Expression(assignment.value)
    assert Path(eval(compile(expression, str(source), "eval"), values)) == expected
