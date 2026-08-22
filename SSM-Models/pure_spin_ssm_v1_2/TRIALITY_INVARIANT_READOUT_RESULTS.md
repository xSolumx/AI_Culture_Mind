# Triality-invariant readout result

**Date:** 2026-08-22
**Verdict:** failed the prospective promotion threshold; direction readout
remains the maintained v1.2 default.

## Result

The candidate appended the three sector log-energies and bounded cubic
triality contraction to each channel while retaining the normalized state
direction. The mechanism is numerically Spin(8)-invariant, scale-sensitive,
finite-gradient, and adds only 4,096 parameters.

The gate and seeds were committed as `34c2ff8` and pushed before training.
Positive improvement means lower candidate validation bits per byte.

| seed | direction bpb | invariant bpb | improvement |
|---:|---:|---:|---:|
| 71 | 2.71058 | **2.69516** | +0.01543 |
| 73 | 2.72186 | **2.72010** | +0.00177 |
| 79 | 2.72772 | **2.72510** | +0.00262 |
| mean | 2.72006 | **2.71345** | +0.00661 |

The candidate won all three seeds and had no regression, but the frozen rule
also required at least +0.0100 bpb mean improvement. It therefore failed one
of four required conditions and is not promoted. The result is evidence that
the old RMS-only interface discards weakly useful information, but the recovered
invariants close only about 2.6% of the previously measured 0.2535 bpb mean gap
to fused Mamba-2. That is too small to justify changing the default.

Sequential training times are retained in the source artifacts but are not a
speed result. The added contractions are outside the fused recurrence kernel;
any future systems claim requires order-balanced timing after a quality gate
passes.

## Consequence

The next candidate should not append more invariant scalars. The larger missing
mechanism is token-dependent content routing across the two equivalent copies
of each triality representation. Programme 1's reducible-isotypic result gives
the exact legal form: an operator on `V tensor R^m` may mix the multiplicity
axis while the shared Spin action acts on `V`. v1.3 independently found that
query-directed memory access matters more than dense exceptional transport.

The next isolated gate is therefore a Schur-legal multiplicity query/router at
readout, leaving the bounded recurrence and raw CUDA scan unchanged. This is a
larger expressivity intervention than amplitude summaries but remains cheap
and mathematically auditable.

## Artifacts

- `artifacts/shakespeare_triality_readout_gate_seed71_cu126_f14a.json`
- `artifacts/shakespeare_triality_readout_gate_seed73_cu126_f14a.json`
- `artifacts/shakespeare_triality_readout_gate_seed79_cu126_f14a.json`
- `artifacts/shakespeare_triality_readout_gate_summary_cu126_f14a.json`

The summary tool rejects missing/wrong seeds, mismatched source hashes,
environment, data, parameters, readout order, or non-finite results before
applying the preregistered decision.
