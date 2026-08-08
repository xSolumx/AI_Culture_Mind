# AI Culture Mind

AI Culture Mind is a research workspace, not one monolithic model claim. Its
work is separated into independently falsifiable programs with different
mathematical objects, evidence standards, and publication paths.

Start with the [research-program index](research-programs/README.md). It
distinguishes:

1. selective rotor state-space models;
2. shared-family representation learning;
3. triality memory and intertwiner scans;
4. Spin(8) sensing and Cayley design;
5. Spin(8) Dirac--Gram inequalities;
6. Spin(9) Dirac--Clifford sensing;
7. controlled model benchmarks;
8. the historical SpinorModel prototype.

The [foundational claim and logic audit](FOUNDATIONAL_CLAIM_AUDIT_2026-08-08.md)
records which mathematical and empirical claims survived an adversarial
definition--domain--evidence review, the corrections made, and the gates that
remain open.

No result in one program is evidence for another unless an explicit bridge
experiment or theorem is cited. In particular, the maintained language model
uses `Cl(3,0)`/`Spin(3)`, whereas the triality mathematics concerns three
eight-dimensional representations of `Spin(8)`.

## Source layout

| Path | Role |
|---|---|
| [`research-programs/`](research-programs/README.md) | Public claim map, status ledgers, and reading order |
| [`SSM-Models/`](SSM-Models/) | Maintained rotor-SSM implementation and transport ablations |
| [`Spin8-Triality-Research/`](Spin8-Triality-Research/) | Standalone theorem repository, linked as a Git submodule |
| [`Spin8-SSM-Benchmark/`](Spin8-SSM-Benchmark/) | Matched empirical benchmarks and controls |
| [`SpinorModel/`](SpinorModel/) | Preserved original prototype and a separate overhaul |

The existing project paths are retained deliberately: they are source and
provenance boundaries, and moving them would invalidate historical links and
artifact manifests. The new program layer reorganizes the scientific claims
without silently rewriting that history.

Large model weights, downloaded datasets, generated caches, and raw process
logs are intentionally excluded from Git. Reproducible conclusions should be
backed by structured artifacts, executable checks, and a concise interpretation
that states both the pass criteria and the nonclaims.
See the [public-release policy](PUBLICATION_SCOPE.md) for the complete boundary.

Clone the repository and its theorem submodule with:

```bash
git clone --recurse-submodules https://github.com/xSolumx/AI_Culture_Mind.git
```
