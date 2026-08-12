# Endpoint-only length curriculum: prospective contract

Date fixed: 2026-08-03, after fixed-length seed 0 failed and before any
curriculum GPU result. Fixed-length replication seeds 1 and 2 were already in
flight; their outcomes were not used to choose this schedule.

## Motivation fixed before results

The endpoint compiler succeeds exactly, but the first fixed-length-16 neural
run remains at chance: loss approximately `log(60)`, no representation trigger,
and final accuracy near 1/60 through 2,000 steps. This isolates neural
optimization, not algebra identifiability, as the failed gate. It does not by
itself distinguish temporal depth from sparse/directionless supervision.

The intervention preserves endpoint-only supervision while shortening the
credit-assignment path early in optimization. Every example remains an
independent complete word with exactly one endpoint label. No example exposes
its own prefix labels, and no batch is expanded into a prefix trace.

Length-1 endpoints nevertheless reveal one-step outcomes. Thus "endpoint-only"
means no internal prefix trace per example; it does not mean the curriculum
withholds all short-transition information.

## Frozen curriculum

| Steps | Complete-word length | Endpoint labels per example |
|---:|---:|---:|
| 1--250 | 1 | 1 |
| 251--500 | 2 | 1 |
| 501--750 | 4 | 1 |
| 751--1000 | 8 | 1 |
| 1001--2000 | 16 | 1 |

Each stage uses an independent deterministic RNG stream derived from training
seed and stage index. The length-1 stage has no bigram split; every later stage
retains the fixed held-out `(0,2)` training bigram. Total neural supervision
remains exactly 512,000 endpoint labels—the same as the failed fixed-length
run and 16 times fewer labels than dense prefix training.

The endpoint compiler remains unchanged: 1,024 passive length-16 endpoint
labels plus 124 active membership queries, 1,148 total. Holonomy begins after
step 750, receives endpoint labels only, and composes them using the already
recovered exact action.

## Frozen cohort and gates

- Seeds 0 through 9.
- Four-channel pure Cl(3) rotor recurrence.
- 2,000 deterministic CUDA updates, batch 256.
- Unconstrained ambient search followed only after detection by joint
  shared-conjugacy retraction.
- Untouched changed-generator index 59.

Every seed must:

1. pass the fixed 1,148-label endpoint compiler contract;
2. trigger representation discovery by step 1,500;
3. compile below `1e-10` invariance and homomorphism RMS;
4. reach at least 90% at every dense L16--L256 checkpoint on original and
   index-59 alphabets;
5. reach at least 90% at L4096 on both and L16384 on index 59;
6. pass the 13-point L4096--L16384 index-59 sweep.

The headline remains the all-seed count. If seed 0 or any later seed fails,
that failure is not repaired by changing stage lengths, trigger threshold,
loss weights, or training budget within this experiment.

## Interpretation boundary

A pass would show that curriculum can replace dense prefix credit assignment
without increasing endpoint-label count. It would not alone distinguish
ordered continuation from short-word signal density; the separately frozen
shuffled-mixture and fixed-L16 persistence controls make that causal split. It
would not be evidence that passive fixed-length endpoints suffice, because the
schedule deliberately supplies short complete words. It also retains exact
anonymous endpoint classes and an active representative-extension oracle, and
therefore does not establish noisy, unlabeled, or natural-language discovery.
