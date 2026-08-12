# Cayley information spectrum paper

This directory contains an arXiv-oriented LaTeX manuscript for the exact
orthonormal balanced-orbit theorem. It intentionally does **not** claim the
open unrestricted Dirac--Gram inequality or global five-query optimality.

The compact, standard-library-checkable review object is the
[`referee package`](../../referee/cayley-information-spectrum/README.md). Read
that package first to see the theorem statement, structural proof, exact
coefficient artifact, and the boundary between what the minimal verifier
recomputes and what it takes as input.

## Render status

The manuscript was compiled with Tectonic 0.17.0 and inspected page by page on
2026-08-07. The verified five-page PDF is generated locally at
`output/pdf/balanced-cayley-information-spectra.pdf`.
The build has no unresolved references, missing citations, overfull boxes, or
underfull boxes.

The self-contained arXiv upload archive is generated locally at
`output/arxiv/balanced-cayley-information-spectra-source.zip`.
It contains only `main.tex` and `references.bib`, uses arXiv-safe file names,
and was compiled successfully after extraction as an isolated smoke test.

Before archival submission:

1. rerun the exact publication tests and artifact manifest verifier;
2. include the repository URL and immutable release identifier in the data and
   code availability section.

Typical build:

```text
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The locally verified equivalent is:

```text
tectonic main.tex --keep-logs --keep-intermediates
```
