# Q8 representation-quality decoder gate: prospective validation contract

Date fixed: 2026-08-03, after observing seeds 0--9 and before training or
evaluating validation seeds 10--19.

## Selection-cohort finding

Joint all-channel retraction makes every seed algebraically exact but preserves
the behavioral gate in 9/10. In seed 5, two channels lie near the Q8 manifold
and decode perfectly alone; two channels require large action repair and become
decoder-misaligned. The pre-retraction-to-exact projection distances across the
40 selection-cohort channels are sharply bimodal:

- representation channels: `0.0004`--`0.0116` RMS;
- nuisance channels: `0.4918`--`0.6977` RMS.

## Frozen rule

Set a channel's decoder columns to zero iff its joint Q8-family projection RMS
exceeds `0.10`. Keep every channel action in the exact jointly retracted Q8
family. Do not use endpoint labels, accuracies, margins, decoder gradients,
channel ablations, or per-seed thresholds. Do not retrain any parameter.

The threshold is deliberately placed in the empty order-of-magnitude gap. It
was selected using seeds 0--9, so those seeds are a selection cohort and cannot
validate the rule.

## Validation

Train fresh seeds 10--19 with the already-frozen spinor curriculum. Report raw,
all-channel-retracted, and quality-gated-retracted outcomes separately.

The prospective rule passes only if all ten validation seeds:

1. retain at least one channel below the `0.10` projection threshold;
2. reach >=99% pair-member and both-member accuracy at every full dense and
   long odd/even checkpoint after gating;
3. retain <=`1e-5` per-channel and whole-model Q8 homomorphism/relator RMS;
4. preserve recurrent state parity within `1e-5`.

Raw or ungated failures remain failures of those variants. A gated pass would
establish reliability of the complete discovery/retraction/quality-selection
pipeline, not 10/10 reliability of unconstrained SGD by itself.
