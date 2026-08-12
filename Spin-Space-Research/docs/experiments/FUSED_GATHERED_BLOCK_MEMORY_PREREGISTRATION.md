# Fused gathered-block memory preregistration

- **Frozen:** 2026-08-10, after the eager gathered result and before fused
  implementation or timing inspection
- **Hardware:** local RTX 2070 SUPER, Windows PyTorch CUDA
- **Dependency:** `triton-windows==3.7.1.post27`
- **Processes:** three new independent measurement processes

## Motivation

The frozen eager benchmark established that physical block gathering beats the
equivalent masked-full implementation and sharply reduces allocation, but loses
to the dense-full PyTorch path because routing, gathering, updating, scattering,
and reading launch many small kernels. This follow-on tests the declared cause
by fusing those operations into one Triton kernel. The eager result and its
decision remain unchanged.

## Fused operation

One program handles one batch element and performs:

1. coarse block scores for the write alias;
2. argmax block selection and eight-way fine softmax;
3. gathered direct or normalized-delta update of the selected `8 x 8` block;
4. coarse and fine routing for the query alias;
5. gathered read, using the newly updated block when write and query select the
   same block.

The operation mutates persistent state in place and is an inference recurrence.
No backward claim is made.

## Frozen grid and measurement

- logical slots: `64`, `256`, `1024`, `4096`;
- slots per block: `8`;
- alias/value dimension: `8`;
- batches: `1`, `16`, `64`;
- update laws: direct and standard normalized delta;
- dtype: `float32`;
- warmup: `100` calls;
- timing blocks: `25`;
- inner calls per timing block: `500`;
- cyclic timing order among eager dense, eager gathered, and fused gathered;
- every raw timing block retained.

## Correctness gates

For every process and cell, one-step fused and eager-gathered execution must
agree within:

- final-state maximum absolute error `<= 1e-5`;
- prediction maximum absolute error `<= 1e-5`;
- all states and predictions finite.

## Frozen decision

Fusion is supported only if the median of the three process medians is faster
than both eager gathered and eager dense for direct and delta updates at batch
`16`, slots `1024` and `4096`. The complete small-batch and large-batch frontier
is reported regardless of the decision.

## Boundary

This can establish an inference-kernel result on the named GPU. It cannot
establish training throughput, model quality, hardware-general thresholds,
additional triality capacity, or superiority over full production sparse-
attention systems.

