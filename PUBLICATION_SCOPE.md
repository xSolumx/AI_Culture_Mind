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
