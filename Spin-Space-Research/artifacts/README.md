# Raw artifacts

This directory contains the JSON outputs that support the research documents.
They are retained as immutable evidence, including failed gates and controls.
The standalone extraction began with 255 source-derived artifacts; subsequent
exact results are added here with the same hash-manifest discipline.

- `ARTIFACTS.sha256` at repository root hashes every JSON file here.
- `PROVENANCE.json` records the 255 source-derived artifacts present at the
  standalone extraction boundary. Post-extraction artifacts are explicitly
  new research outputs; `ARTIFACTS.sha256` and Git history provide their byte
  identity and chronology.
- Checkpoints and transient logs are deliberately excluded. Historical
  watchdog records live under `runtime/`, outside this publication set, and
  the twenty-output endpoint-octet campaign has a resumable exact replay tool;
  see [`docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md).

Artifact presence does not upgrade the corresponding claim. Read the matching
preregistration and results document under `docs/experiments/` for the gate,
known confounds, corrections, and interpretation boundary.
