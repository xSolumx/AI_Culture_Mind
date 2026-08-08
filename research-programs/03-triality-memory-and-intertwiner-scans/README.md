# Program 03: Triality memory and intertwiner scans

## Object

Associative recurrent lifts whose transition algebra contains equivariant
bilinear coupling between representation streams. `Spin(8)` triality supplies
one exceptional instance; the broader mathematical object is the triangular
intertwiner scan.

## Claims that currently survive

- A staged triangular bilinear lift can preserve associative prefix scanning
  and constant recurrent state.
- The maintained ordered scan now has a linear-work implementation with
  irregular-length, long-horizon, and full-gradient parity. On the local RTX
  2070 SUPER, its length-4,096 eager-PyTorch forward pass replicated a 3.59x
  speedup over the earlier Hillis--Steele tensor program. This is not a fused
  production-kernel comparison.
- The octonionic/triality binding map is an exact norm-preserving single-pair
  bind/unbind primitive under the stated normalization.
- Multiplicity channels give exact orthogonal slot isolation for at most the
  multiplicity dimension; raw eight-dimensional superposition does not.

## Open empirical question

The program has not established superiority over direct slots, bilinear fast
weights, delta-rule memories, p-BIM, or Householder-product transports under
matched state and compute.

## Canonical evidence

Use the theorem submodule's
[`program ledger`](../../Spin8-Triality-Research/programs/triality-memory/README.md).
The implementation protocol and raw benchmark hashes are in
[`INTERTWINER_SCHURSCAN_BENCHMARK_RESULTS.md`](../../Spin8-Triality-Research/docs/experiments/INTERTWINER_SCHURSCAN_BENCHMARK_RESULTS.md).

## Next publishable question

Present Intertwiner SchurScans as the general theorem, with triality as an
example. The next systems step is a fused direct-affine scan kernel; the next
scientific step is a narrowly matched retrieval benchmark that can distinguish
the equivariant lift from a same-width generic bilinear recurrence.
