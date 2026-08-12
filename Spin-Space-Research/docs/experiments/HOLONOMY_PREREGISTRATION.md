# Path-Holonomy Experiment Preregistration

Date: 2026-08-02

## Hypothesis

The 2.2-radian Cl(3) rotor chart removes training-length convergence collapse,
but long-word errors persist because locally small transition defects can add
coherently along particular paths. Aligning complete alternate paths to a
canonical representative of the same A5 element should reduce that coherent
drift more directly than mean or tail one-step Cayley closure.

## Known degenerate solution

Holonomy is not an anti-collapse objective. If every token action is identity,
every word and every closed loop produces the same state, so path-holonomy loss
is exactly zero. A margin term by itself detects this collapse but has zero
first derivative at perfectly coincident states when implemented with squared
distance. Therefore the experiment must never run holonomy or margin alone.

The non-collapse contract is:

1. ordinary supervised prefix cross-entropy remains active with unit weight;
2. canonical and alternate-path states receive an explicit class-separation
   margin;
3. the tangent-at-identity test must continue to show nonzero action gradients;
4. final evaluation must retain nonzero prototype margin, action displacement,
   and generator commutator separation.

An all-identity or non-faithful quotient solution fails this contract even if
its holonomy loss is zero.

## Path construction

Each holonomy path concatenates four independently shifted length-16 training
words, producing a length-64 word without introducing a new token
distribution. Its target group element is composed from the four supervised
segment targets. The learned length-64 operator is compared with the learned
canonical short-word operator for that same element on the structurally active
input sector:

- Cl(3) rotor: the 24 vector/bivector coordinates across four channels;
- Householder: all 32 coordinates.

The loss uses a power-8 aggregate over path/channel errors to emphasize the
tail while preserving squared-error units. This is a benchmark upper-bound
objective because A5 labels and canonical representatives are available.

## Separation term

The separation loss contains both:

1. a hinge margin over all 60 canonical orbit prototypes; and
2. a triplet margin requiring every alternate-path state to be closer to the
   canonical prototype of its own target than to any other target prototype.

This explicitly includes non-canonical alternate paths; canonical margin alone
is not accepted as sufficient protection against alternate-path collapse.

## Predeclared pilot

- family: `pure_ga_rotor`
- seed: 2, chosen before this experiment because capped seed 2 converges at
  L16 but fails long retention
- rotor cap: 2.2 radians
- task length: 16
- holonomy path length: 64 (four segments)
- holonomy batch: 64
- holonomy power: 8
- holonomy weight: 0.01
- separation margin: 0.5
- separation weight: 0.1
- start step: 750
- linear ramp: 500 steps
- total steps: 1,500

## Acceptance and rejection criteria

The pilot is promising only if all of the following hold relative to capped
seed-2 control:

1. L16 final-position accuracy remains at least 95%;
2. both L64 and L128 improve, with L128 exceeding the control's 78.8%;
3. length-64 path-holonomy RMS falls materially;
4. canonical and alternate-path minimum/nearest-negative margins remain
   positive;
5. canonical decoded accuracy remains 100%;
6. action displacement and commutator separation remain nontrivial;
7. state-norm and streaming-equivalence checks continue to pass.

A lower holonomy loss without improved L128 is a rejection, as are identity
collapse, prototype collapse, or a decoder-only success with nonpositive state
margin. No multi-seed claim will be made from this selected-seed pilot.

## Preregistered seed-2 outcome

The pilot passes every acceptance criterion:

| Metric | capped control | path holonomy |
|---|---:|---:|
| L16 final accuracy | 100% | 100% |
| L64 final accuracy | 80.9% | 99.9% |
| L128 final accuracy | 78.8% | 99.1% |
| length-64 path RMS | 1.0345 | 0.6753 |
| alternate target distance | 2.0222 | 0.6719 |
| nearest-negative squared-distance margin | -3.6769 | +0.2979 |
| orbit-directional homomorphism RMS | 0.6099 | 0.2694 |
| canonical prototype margin | 0.8965 | 0.8963 |
| decoded canonical products | 100% | 100% |

Action displacement (`0.4328`) and generator commutator separation (`0.3110`)
remain nontrivial. The global worst operator residual stays near `1.41`, so the
gain does not come from repairing every state-space direction. It comes from
aligning the reachable long paths and moving alternate words inside the
correct decoder-separated state regions. This is the behavior the hypothesis
predicted.

The selected-seed pilot is a mechanism success, not yet a reliability claim.
The next registered action is replication on the other capped-GA retention
failures (seeds 5, 6, 7, and 9) plus successful seed 0 as a harm check.

## Ten-seed replication

After the selected stress set, the unchanged objective was run on every seed:

| Family / objective | complete functional gate | positive-margin state contract | mean L64 | mean L128 |
|---|---:|---:|---:|---:|
| GA, 2.2-radian cap | 5/10 | not measured | 90.0% | 85.2% |
| GA, cap + path holonomy | **8/10** | 7/10 | **94.5%** | **97.5%** |
| two Householders | 8/10 | not measured | 94.1% | 89.6% |

All holonomy runs retain 100% L16 accuracy. Seeds 0, 1, 2, 3, 5, 8, and 9
pass both long-length thresholds with positive alternate-path margin. Seed 7
is functionally perfect at L64/L128 but has a slightly negative alternate-path
margin (`-0.0462`), so it fails the stricter state contract. Seed 4 reaches
`96.0%/75.6%`; seed 6 reaches `49.3%/100%`. These non-monotone failures show
that a fixed length-64 holonomy distribution can leave length-selective phase
aliases even when L128 improves.

The replicated claim is therefore: path holonomy raises the functional gate
from 5/10 to 8/10 and substantially raises mean long-length accuracy without
training-length collapse. It matches Householder's functional pass count and
exceeds its mean L128 accuracy in this experiment. It does not yet provide
uniform-in-length state separation.

### Gate hierarchy and distribution

"Functional gate" is deliberately not the original mechanism gate:

1. **Original raw-operator mechanism gate:** homomorphism RMS at most `1e-3`.
   No data-trained holonomy run approaches this; the result is 0/10. Only the
   oracle-irrep construction has passed it.
2. **Functional gate:** at least 95% at L16 and at least 90% at both L64 and
   L128. Holonomy passes 8/10.
3. **State-geometry contract:** the functional gate plus positive alternate-
   path margin and the non-collapse diagnostics. Holonomy passes 7/10 because
   functionally perfect seed 7 has margin `-0.0462`.

The per-seed distribution must accompany the means:

| Seed | L64 | L128 | alternate margin | functional | state contract |
|---:|---:|---:|---:|:---:|:---:|
| 0 | 100% | 100% | +0.302 | pass | pass |
| 1 | 100% | 100% | +0.276 | pass | pass |
| 2 | 99.9% | 99.1% | +0.298 | pass | pass |
| 3 | 100% | 100% | +0.334 | pass | pass |
| 4 | 96.0% | 75.6% | +0.082 | fail | fail |
| 5 | 100% | 99.8% | +0.230 | pass | pass |
| 6 | 49.3% | 100% | +0.138 | fail | fail |
| 7 | 100% | 100% | -0.046 | pass | fail |
| 8 | 100% | 100% | +0.320 | pass | pass |
| 9 | 100% | 100% | +0.284 | pass | pass |

The `94.5%/97.5%` L64/L128 means summarize a sharply non-Gaussian result:
eight seeds are near-perfect, while seeds 4 and 6 fail in different,
non-monotone ways. The median is 100% at both lengths. Means are never to be
reported without this failure distribution.

### Hyperparameter provenance

Holonomy weight `0.01`, margin weight `0.1`, margin target `0.5`, power `8`,
start step `750`, ramp `500`, and four-segment length-64 construction were fixed
after the seed-2 pilot and before inspecting seeds 5, 6, 7, and 9. They were
then left unchanged for the seed-0 harm check and seeds 1, 3, 4, and 8. No
per-seed weight or schedule tuning occurred.

## Preregistered multi-scale follow-up

The fixed-scale failures are seed 4 (`96.0%/75.6%`) and seed 6
(`49.3%/100%`) at L64/L128. Before changing the implementation, the follow-up
is restricted to those two known failures and uses the same loss weights,
margin, start step, ramp, optimizer, and data. The only change is cycling one
holonomy segment multiplier per active step through `2, 3, 4, 5`, corresponding
to path lengths 32, 48, 64, and 80.

The hypothesis is that a coherent angular defect can alias at one fixed path
length but cannot cancel consistently across coprime segment counts. The
follow-up passes only if both seeds retain at least 95% at L16, exceed 90% at
both L64 and L128, and have positive alternate-path nearest-negative margin at
every evaluated holonomy scale. Improvement at only one horizon is rejection.
These are selected-seed repair experiments, not a new reliability sample.

The dense L16-L256 sweep was requested in the command, but a whole-sweep floor
was not written into the original follow-up criterion. It is therefore
exploratory for seed 4 and must not be relabeled as preregistered. Before seed 6
completed, a secondary prospective interpretation was recorded for that
remaining run: minimum accuracy of at least 90% at every sampled length means
uniform alias suppression; any interior value below 90% means the alias moved,
even if L64 and L128 pass.

## Multi-scale selected-seed outcome

Both seeds pass the original written follow-up criterion:

- seed 4 reaches 100% at L64/L128 and has positive margin at all four scales;
- seed 6 reaches 97.5%/100% at L64/L128 and also has positive margin at all
  four scales.

The exploratory/prospective dense interpretation separates them:

- seed 4 is uniformly rescued: its minimum across sampled L16-L256 lengths is
  99.8% (at L160);
- seed 6 is not uniformly rescued: it falls to 81.2% at L48, 78.3% at L112,
  and 87.1% at L224, despite reaching 100% at L128.

Thus coprime multi-scale holonomy eliminates the observed alias in seed 4 but
moves rather than eliminates it in seed 6. These selected repairs do not change
the ten-seed headline. A full unchanged multi-scale run is required for a new
reliability claim.

## Completed ten-seed dense follow-up

The full unchanged run is complete. It passes 10/10 at the named checkpoints
and 10/10 on positive margins at all four sampled holonomy scales, but only
8/10 on the dense at-least-90% L16-L256 floor. Seed 6 reaches a minimum of
76.37% at L176, and checkpoint-clean seed 9 reaches 87.50% at L240. Median
dense minimum is 97.22%. The dense protocol is now the default, but it remains
a finite-horizon characterization rather than an exact-mechanism certificate.

The subsequent post-hoc channel audit localized these failures to nuisance
channel logits rather than the learned oracle-like channel. See
`OBSERVABLE_IRREP_RESULTS.md`; this later causal result is explicitly not part
of the preregistered multi-scale criterion.
