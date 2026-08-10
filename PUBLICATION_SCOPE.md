# Public-release policy

The parent repository publishes maintained source, reproducible structured
artifacts, and documents whose evidence boundary is explicit.

## Public

- source and tests needed to reproduce a maintained implementation;
- complete JSON or Markdown result artifacts with a validation path;
- exact theorem code delegated to the pinned theorem submodule;
- negative or inconclusive results when their protocol and limitations are
  preserved;
- program charters that separate claims which require different evidence.

## Not public, or not public yet

- credentials, private contact details, and machine-specific absolute paths;
- model checkpoints and third-party data without a deliberate distribution
  decision;
- caches, temporary renders, raw logs, profiler dumps, and crash files;
- private reviewer conversations or unedited model commentary;
- incomplete benchmark artifacts, smoke outputs, and OOM traces presented as
  if they were results;
- raw proof-search grids whose deterministic generator and compact certificate
  are the appropriate public objects.

Local exclusion is intentional. It does not strengthen any public claim.

## Benchmark publication contract

A systems number is publishable only with a frozen selection rule, complete
machine-readable artifact, validation path, software and hardware metadata,
warm-up policy, sample counts, and peak-memory method. Kernel timing must be
measured in fresh processes after tuning, and ideal payload bytes must not be
reported as measured CUDA allocation. State-, parameter-, token-, and
measured-compute matching answer different questions and must be labelled
separately.

The 2026-08-10 hierarchical-memory campaign follows this boundary. It reports
an official FLA chunk-kernel speed result for the co-moving delta formulation,
but does not promote that result to language-model quality, triality capacity,
or a production sparse-router latency claim.
