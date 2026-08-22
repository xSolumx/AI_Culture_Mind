# Shared-action compression gate

**Frozen before training:** 2026-08-22

## Candidate

The recurrent multiplicity controller failed its valid paired gate, but the
compiler exposed a smaller independent question. Established v1.2 learns a
separate Spin coordinate vector for each of its two recurrent channels. The
candidate ties those coordinates:

\[
H_{t,c}=s_{t,c}H_{t-1,c}R_t^T+D_{t,c},\qquad c\in\{1,2\},
\]

using one shared `R_t` and no multiplicity rotation. The channel retentions and
writes remain independent. This is the `coupled_isotypic` semantic family at
`recurrent_multiplicity=identity`, lowered by `raw_cuda_coupled`.

Sharing removes 6,708 trainable parameters: 619,808 versus 626,516 for the
established independent-action model, a 1.071% reduction. More importantly for
this GPU, one coordinate stream and one packed warp replace two coordinate
streams and the isotypic-forward schedule. The quality and speed questions are
kept separate.

This candidate is not mathematically equivalent to established v1.2 and exact
tensorwise paired initialization is not claimed. Runs share the same seed,
training batches, validation windows, optimizer, and data. The quality decision
therefore uses three fresh seeds.

## Quality non-inferiority gate

- baseline: independent recurrence, `raw_cuda_hybrid`;
- candidate: shared-action identity recurrence, `raw_cuda_coupled`;
- seeds: `179`, `181`, and `191`;
- 300 steps, batch 8, sequence length 256, 16 fixed validation batches;
- Tiny Shakespeare raw bytes, maintained 90/5/5 split;
- group schedule `(3,4,6,8)`, direction readout, SwiGLU, no optional routers;
- maintained PyTorch 2.10 + cu126 WSL environment on the RTX 2070 SUPER.

Positive improvement is `independent bpb - shared-action bpb`. Quality is
non-inferior only if all hold:

1. mean improvement is at least `-0.0100` bpb;
2. at least one of three seeds is non-regressive;
3. no seed regresses by more than `0.0500` bpb;
4. all values and gradients are finite and artifact hashes agree.

This is deliberately a non-inferiority rule: the candidate offers parameter
and potential execution savings rather than extra expressivity. A quality pass
does not promote it by itself.

## Conditional throughput gate

Only a quality pass authorizes an order-balanced fixed-batch campaign. It must
use complete forward/backward/gradient-clipping/AdamW steps, alternating model
order across at least four cycles, with 10 warmup steps and five windows of 10
steps per model per cycle. Promotion requires the median of cycle medians to be
at least 5% faster for the shared-action model and requires a confirmatory
repeat whose ordering agrees. Embedded sequential training timers are not used
for this decision.

If quality fails, shared action remains a correct compiler control rather than
a maintained model. If quality passes but speed fails, it remains an optional
compression control. Neither outcome changes the negative recurrent-mixing
decision in `COUPLED_ISOTYPIC_RESULTS.md`.
