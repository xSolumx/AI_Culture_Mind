# Program 07: Controlled model benchmarks

## Object

Matched-path empirical evaluation of proposed Spinor/SSM models against direct,
delta-rule, Mamba-family, and other appropriate baselines.

## Contract

- A run is a result only when its structured artifact is complete and passes
  validation.
- Matching state, parameters, and measured compute answer different questions;
  all must be labeled.
- An out-of-memory event, partial log, or smoke test is operational evidence,
  not a model-quality result.
- The benchmark repository must not retroactively strengthen theorem claims in
  the Spin8 submodule.

## Canonical location

- [`Spin8-SSM-Benchmark/`](../../Spin8-SSM-Benchmark/)

## Next publishable question

The memory-core matrix is complete: frozen ten-seed routing cohorts and fresh-
process CUDA measurements now compare dense, block, and hard routing plus
direct, native delta, co-moving FLA chunk, and recurrent transported memory.
The next publishable question is the **model-level** transfer: insert the
hierarchical co-moving memory into one controlled sequence model and compare it
against strong attention/delta/SSM baselines with fully validated artifacts,
identical tokens, and separate state/parameter/compute tables. The completed
memory-core benchmark must not be described as that model result.
