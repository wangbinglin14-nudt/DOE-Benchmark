"""Locate organized working files; accept legacy flat test/release layouts."""
from pathlib import Path

def paper_file(root: Path, name: str) -> Path:
    root = Path(root)
    name = str(name)
    if name.startswith("si_"):
        group = "tables"
    elif name.startswith("preregistration"):
        group = "protocol"
    elif name in {"cover_letter.md", "significance_statement.md", "reporting_summary.md", "SUBMISSION_READINESS.md"}:
        group = "editorial"
    elif name in {"nature_cs_main.pdf", "supplementary.pdf"}:
        group = "pdf"
    else:
        return root / name
    return root / group / name if (root / group).is_dir() else root / name

def metadata_file(root: Path, name: str) -> Path:
    root=Path(root)
    return root / "metadata" / name if (root / "metadata").is_dir() else root / name

def project_file(root: Path, relative: str) -> Path:
    root=Path(root);relative=Path(relative)
    if len(relative.parts)==1 and relative.name in {'.zenodo.json','codemeta.json','croissant-metadata.json','ro-crate-metadata.json'}:
        return metadata_file(root,relative.name)
    if len(relative.parts)==2 and relative.parts[0]=='paper':
        return paper_file(root/'paper',relative.name)
    return root/relative
