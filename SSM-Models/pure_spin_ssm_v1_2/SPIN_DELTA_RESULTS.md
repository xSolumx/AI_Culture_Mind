# Spin-Delta frozen gate results

**Decision:** failed; do not promote Spin-Delta as the v1.2 successor.

## Valid replacement cohort

The pairing amendment froze seeds 373/379/383 and a `2e-6` initial-logit
compatibility bound before training. All three artifacts have bitwise-identical
common parameters, matching implementation and dataset hashes, finite losses,
and pairing residuals below `1e-6`.

| Seed | Maintained bpb | Spin-Delta bpb | Improvement | Win |
|---:|---:|---:|---:|:---:|
| 373 | 2.68618428 | 2.67825278 | +0.00793150 | yes |
| 379 | 2.72452527 | 2.78391485 | -0.05938958 | no |
| 383 | 2.66948402 | 2.69364772 | -0.02416370 | no |

Positive improvement is maintained minus Spin-Delta. The candidate won one of
three seeds, averaged `-0.02520726` bpb, and had a worst regression of
`-0.05938958` bpb. It therefore failed three frozen requirements:

- wins: 1, required at least 2;
- mean improvement: `-0.02520726`, required at least `+0.0100`;
- worst regression: `-0.05938958`, allowed no worse than `-0.0500`.

Compatibility and finiteness passed. The aggregate decision in
`artifacts/spin_delta_gate/summary.json` is `quality_pass=false` and
`speed_gate_authorized=false`.

## Commissioning cohort boundary

The original seed-353 result is not part of the decision. It completed before
seed 359 exposed an over-tight numerical pairing guard. Because its favorable
`+0.01268566` bpb result was already visible, the entire original cohort was
quarantined rather than changing a threshold around a known outcome. The
reason, artifact hash, corrected bound, and replacement-seed pairing probes are
recorded in `SPIN_DELTA_PAIRING_AMENDMENT_2026-08-22.md`.

## Systems observations, not a speed gate

The sequential quality-run timers placed Spin-Delta at 0.699x, 0.680x, and
0.588x maintained throughput, with a mean ratio of 0.655x. Peak allocated CUDA
memory was 168,242,688 bytes versus 156,825,088 bytes maintained. These are
diagnostics from fixed variant ordering, not an order-balanced speed result.
Because quality failed, no optimization or speed gate is authorized for this
candidate.

## Interpretation

The result rejects the specific hypothesis that one extra addressable slot per
independent Spin head, under this smooth staged initialization and 300-step
budget, supplies the missing language-model operation. The isolated seed-373
win and invalid seed-353 pilot do not overcome the two valid regressions.

This does not invalidate the compiler. The `(batch, head)` grid lowering,
two-slot contractive affine monoid, semantic parallel scan, and raw-CUDA
full-gradient backend remain verified reusable components. It also does not
falsify delta memory in other parameterizations, longer training, or tasks
with explicit retrieval demands. Those are new hypotheses and require new
preregistration; they cannot rescue this gate retroactively.
