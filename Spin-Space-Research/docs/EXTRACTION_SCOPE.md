# Extraction scope audit

## Commit boundary

The public research repository was extracted from
`xSolumx/AI_Culture_Mind` over the closed change interval

```text
ce727caf2c7e8ed827447ad12a32704a62f91a99
    ..
a367a80e291d78b2b1db694f764acaa4b5c98ceb
```

The upper endpoint is the frozen exact signed Dirac-star theorem commit.

## Complete scientific payload

The interval contains 826 added or modified paths under `SSM-Models`. The
extraction audit classifies them as follows:

| Published category | Files |
|---|---:|
| Python research source | 79 |
| Python tests | 4 |
| Analysis and launch tools | 3 |
| Experiment preregistrations and result documents | 112 |
| Foundation/review/program documents | 7 |
| Raw JSON artifacts | 255 |
| Dependency specifications | 2 |
| Tiny controlled corpus | 1 |
| **Published source-derived files** | **463** |

The dependency audit found zero missing local Python modules in the extracted
source closure.

## Generated payload exclusions

The remaining 363 paths are generated or repository-local payload:

| Excluded category | Files | Reason |
|---|---:|---|
| Model/compiler checkpoints | 301 | Generated binary state; reproducible claims use JSON gates |
| stdout/stderr and transient logs | 60 | Ephemeral execution output |
| Historical language-model checkpoint | 1 | Unrelated 44.8 MB predecessor model |
| Parent-repository `.gitignore` | 1 | Replaced by repository-specific publication metadata |

No preregistration, results document, source harness, test, raw JSON result, or
negative finding was excluded.

## Layout modifications

`PROVENANCE.json` records both the original source hash and published hash.
Raw JSON artifacts are byte-preserved. Source and document files are normalized
to LF for a clean cross-platform repository, and `modified_for_layout=true`
records that publication-level normalization together with the following path
adjustments:

- artifact paths in four result documents;
- default artifact output paths in four Spin8 harnesses;
- the relocated theorem artifact path in one test;
- the two portable PowerShell launchers.

No result values or theorem coefficients were rewritten.
