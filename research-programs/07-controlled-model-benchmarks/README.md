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

Complete one small, reproducible benchmark matrix with fully validated JSON,
explicit hardware/runtime metadata, and parameter/state/compute tables before
adding new datasets or architectures.
