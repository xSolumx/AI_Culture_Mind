# Provenance and historical-reading policy

This repository contains both an extracted research record and later work. The
two layers must not be confused.

## The extraction snapshot

`PROVENANCE.json` records 463 files extracted from the source repository through
commit `a367a80e291d78b2b1db694f764acaa4b5c98ceb`. For each file it stores the
source path, published destination, byte count, and SHA-256 values at extraction
time. Those hashes are a historical snapshot; they are not a claim that every
destination still has the same bytes after subsequent research and editorial
work.

The initial publication commit and the source range named in
`PROVENANCE.json` preserve the exact extraction. Later commits record every
post-extraction change. `ARTIFACTS.sha256`, by contrast, is the live integrity
manifest for published raw artifacts and must match the current bytes.

## Current reconciliation

On 2026-08-06, the 463 provenance destinations were checked against their
extraction hashes:

- 441 still matched byte for byte;
- 22 had documented post-extraction changes in Git;
- none were missing.

The changed set consists of current synthesis documents, dated reports that
received explicit corrections or addenda, and one foundational regression
test. Their original bytes remain recoverable from the initial publication
commit and from the source commit named above.

During this manuscript audit, `artifacts/final_summary.json` was found to have
acquired one trailing newline after extraction. Its content had not changed,
but its raw SHA-256 no longer matched either `PROVENANCE.json` or
`ARTIFACTS.sha256`. The file was restored from the initial publication blob;
all 305 current artifact-manifest entries then matched. The verifier now also
rejects any JSON artifact that exists outside the manifest.

## How corrections are recorded

The archive follows four rules:

1. A frozen preregistration remains frozen. Later criteria receive a new name
   and date.
2. A dated result keeps its observations. If its interpretation changes, a
   dated correction or addendum states what changed.
3. Current manuscripts use the corrected interpretation directly and cite the
   historical route only where it is scientifically relevant.
4. Raw artifacts are immutable evidence. Formatting tools must not rewrite
   them, including by adding a terminal newline.

This is the practical meaning of “do not silently rewrite history.” It does not
mean preserving a known false statement as current truth. It means preserving
the original statement, identifying its date and evidence, and making the
later correction equally visible.

For present theorem status, use the repository [README](../README.md), the
[research map](RESEARCH_MAP.md), and the
[correction ledger](RESEARCH_AUDIT_AND_NEXT_STRATEGY_2026-08-06.md). For the
chronological experiment layer, use the
[experiment-record policy](experiments/README.md).
