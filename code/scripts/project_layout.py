"""Directory configuration and artifact locations for the published repository."""

from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = CODE_ROOT.parent
MANUSCRIPTS_DIR = REPOSITORY_ROOT / "manuscripts"
FIGDATA_DIR = MANUSCRIPTS_DIR / "figdata"
FIGURES_DIR = MANUSCRIPTS_DIR / "figures"
TABLES_DIR = MANUSCRIPTS_DIR / "tables"


def portable_path(path: Path, root: Path) -> str:
    """Keep artifact paths relative to code, including sibling manuscripts."""
    resolved = Path(path).resolve()
    root = Path(root).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        pass
    try:
        return (Path("..") / resolved.relative_to(root.parent)).as_posix()
    except ValueError:
        return resolved.as_posix()


def paper_file(root: Path, name: str) -> Path:
    root = Path(root)
    name = str(name)
    if name.startswith("si_"):
        group = "tables"
    elif name.startswith("preregistration"):
        group = "protocol"
    elif name in {
        "cover_letter.md",
        "significance_statement.md",
        "reporting_summary.md",
        "SUBMISSION_READINESS.md",
    }:
        group = "editorial"
    elif name in {"nature_cs_main.pdf", "supplementary.pdf"}:
        group = "pdf"
    else:
        return root / name
    return root / group / name if (root / group).is_dir() else root / name


def metadata_file(root: Path, name: str) -> Path:
    root = Path(root)
    return root / "metadata" / name if (root / "metadata").is_dir() else root / name


def project_file(root: Path, relative: str) -> Path:
    root = Path(root)
    relative = Path(relative)
    if len(relative.parts) == 1 and relative.name in {
        ".zenodo.json",
        "codemeta.json",
        "croissant-metadata.json",
        "ro-crate-metadata.json",
    }:
        return metadata_file(root, relative.name)
    if len(relative.parts) == 2 and relative.parts[0] == "paper":
        return paper_file(root / "paper", relative.name)
    return root / relative
