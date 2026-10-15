# Compile the current manuscript

Editable sources live in `01_manuscripts/` at the workspace root, also accessible through the code directory's `paper/` junction. Tables are in `tables/`, figures in `figures/pdf/`, and compiled documents in `pdf/`.

Enter `01_manuscripts/` and run:

```sh
pdflatex -output-directory=pdf -interaction=nonstopmode -halt-on-error nature_cs_main.tex
pdflatex -output-directory=pdf -interaction=nonstopmode -halt-on-error nature_cs_main.tex
pdflatex -output-directory=pdf -interaction=nonstopmode -halt-on-error supplementary.tex
pdflatex -output-directory=pdf -interaction=nonstopmode -halt-on-error supplementary.tex
pdflatex -output-directory=pdf -interaction=nonstopmode -halt-on-error supplementary.tex
```

Use pdfLaTeX with geometry, booktabs, graphicx, hyperref, microtype, amsmath, amssymb, lineno, natbib, array, longtable, float and caption. References are embedded in `thebibliography`; no external bibliography database is required. The review manuscript uses 12 pt body text and continuous line numbers.

For Overleaf, preserve the complete manuscript directory structure and choose the main or supplementary TeX source as the main document. The historical filename `nature_cs_main.tex` does not determine the target journal.

Edit sources, not generated PDFs. Numerical changes require regenerating the relevant analysis, tables and figures. There is currently no active submission bundle; the former bundle builder has been archived as a historical tool.

`python scripts/manuscript_source_manifest.py` checks the current source dependencies without duplicating the manuscript tree. After author metadata and availability statements are final, append `--zip-output PATH.zip` to create a source ZIP for upload. The command refuses to overwrite an existing ZIP. This ZIP is only the manuscript source, not the research code/data release.
