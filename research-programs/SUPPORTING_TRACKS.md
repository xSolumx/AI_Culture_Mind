# Supporting evidence and provenance tracks

These are not additional numbered research programmes.

## Controlled model benchmarks

[`Spin8-SSM-Benchmark/`](../Spin8-SSM-Benchmark/) contains matched-path
empirical evaluation against direct, delta-rule, Mamba-family, and other
appropriate baselines. A run is a result only when its structured artifact is
complete and validated. State-, parameter-, and compute-matched comparisons
answer different questions and must stay separately labelled. Out-of-memory
events, partial logs, and smoke tests are operational evidence only.

The next model-level gate is an FLA-compatible selected-block mixer with
recurrent cache, differentiable reference/backward, and fused-inference
contracts, followed by a controlled sequence-model comparison. Existing
memory-core benchmarks are not that model result.

## Historical SpinorModel prototype

[`SpinorModel/`](../SpinorModel/) preserves the original compact
geometric-algebra prototype and its separate
[`overhauled/`](../SpinorModel/overhauled/) experiment. It is implementation
lineage, not the maintained model contract. Historical checkpoints retain
their original tensor and target conventions; overhaul results do not replace
claims about the original model.

The maintained rotor recurrence lives in [`SSM-Models/`](../SSM-Models/) and
is governed by Programme 06.
