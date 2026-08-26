# V1.3.1 quality and learnability results

**Date:** 2026-08-26

**Hardware:** NVIDIA GeForce RTX 2070 SUPER, exact compute capability 7.5

**Verdict:** exceptional actions learn when the task requires them; the repaired
language model learns ordinary text, but official fused Mamba-2 wins every
matched seed in quality, throughput and peak allocation

## What changed before retesting

The older v1.3 screens initialized retention to `sigmoid(2)=0.8808`. An
untouched state therefore retained only about `0.8808^64 = 0.00030` after 64
tokens. They were valid short optimization trajectories but weak memory tests.

Version 1.3.1 changes the default model law:

- initial retention `0.9995` (half-life about 1,386 tokens);
- an independent bounded write-strength gate;
- tied contractive Delta erase by default;
- normalized queries;
- explicit noncompact E6 norm compensation;
- exact bounded-memory direct recurrence beyond the semantic parallel-scan
  length boundary;
- named G2 and Spin(7) rungs below the existing Spin(8)/Spin(9)/F4/E6 ladder.

Historical settings remain reproducible from their commits and artifacts; the
new cohort does not rewrite their conclusions.

## Hidden-coordinate exceptional learning

Each action token hides a signed primitive generator. Training uses words of
length one followed by length four and random Albert probes. Evaluation uses
unseen probes and compositions through length 16. Target directions are chosen
to lie maximally outside the predecessor algebra.

| Target | Predecessor control mean L16 relative error | Correct action mean L16 relative error | Correct passes | Maximum cubic error |
|---|---:|---:|---:|---:|
| F4 | Spin(9): `0.103508` | F4: `0.002071` | 3/3 | `1.64e-13` |
| E6(-26) | F4: `0.130968` | E6: `0.001292` | 3/3 | `2.03e-13` |

Seeds are 2611, 2617 and 2621. The correct arm passes every frozen `0.005`
length-16 gate; every predecessor remains above `0.09`. This establishes that
the real action coordinates and longer compositions can be learned, and that
the extra F4/E6 directions are functionally identifiable.

It does **not** establish autonomous event detection, generic memory,
language quality, or a benefit from using exceptional transport when the data
does not contain the matching action. The fail-closed summary is
[`artifacts/exceptional_learning_summary_sm75_2026-08-26.json`](artifacts/exceptional_learning_summary_sm75_2026-08-26.json).

## Matched natural-text cohort

Every arm used raw UTF-8 Tiny Shakespeare bytes, eager FP32, AdamW at `3e-3`,
1,000 updates, batch 4, length 128, 512,000 scored training targets, 32 fixed
validation batches, and identical target digests within each seed.

| Arm | Parameters | Mean validation bpb | Seed wins vs Mamba-2 | Geometric mean train bytes/s | Maximum peak CUDA allocation |
|---|---:|---:|---:|---:|---:|
| official fused Mamba-2 | 40,848 | **2.78981** | reference | **37,396.5** | **34,941,952 B** |
| safe E6 | 40,858 | 2.94846 | 0/3 | 10,293.6 | 143,826,432 B |
| matched F4 | 40,487 | 2.95962 | 0/3 | 10,473.2 | 141,293,568 B |
| matched identity | 40,722 | 2.97895 | 0/3 | 16,407.9 | 47,430,656 B |

Seeds are 2633, 2647 and 2659. Mamba-2 wins every seed. Against the strongest
exceptional arm, it is better by `0.15865` mean bpb, approximately `3.63x`
faster, and uses about one quarter of the peak allocation.

Safe E6 is the best non-Mamba arm and beats the parameter-matched identity
bundle in all three seeds by `0.03050` mean bpb. Because model widths differ
to achieve parameter matching (`36` identity, `33` F4, `32` E6/Mamba), this is
an architecture-bundle result, not isolated proof that E6 transport caused the
gain. The promotion gate fails. See
[`artifacts/quality_cohort_summary_sm75_2026-08-26.json`](artifacts/quality_cohort_summary_sm75_2026-08-26.json).

## SM75 systems status

- `torch.compile(fullgraph=True, mode="default")` executes the repaired fixed-
  shape E6 model. A 20-update B2/L64 smoke reached 4,915 training bytes/s after
  a 48.6-second preparation. This is execution evidence, not a matched speed
  promotion.
- `reduce-overhead` remains identity-only. Exceptional `matrix_exp` fails CUDA
  graph capture on the current SM75 runtime, and the harness now rejects that
  combination before training.
- The direct recurrent backend is algebraically and gradient-equivalent to the
  dense-transition recurrence. It removes the Hillis-Steele tower of right-
  action prefixes and is the bounded-memory path for lengths above 256. It is
  a correctness backend pending a custom CUDA forward/backward kernel.
- Exact-SM75 forward/backward smokes with one small E6 layer complete at both
  L2048 and L4096 with finite logits. Peak allocation is 322,724,864 and
  627,336,192 bytes respectively. These are single execution/allocation checks,
  not throughput or scaling claims. See
  [`artifacts/long_context_execution_sm75_2026-08-26.json`](artifacts/long_context_execution_sm75_2026-08-26.json).
- The current eager profile and corrected compact-action VJP timings are in
  [`artifacts/profile_v131_sm75_2026-08-26.json`](artifacts/profile_v131_sm75_2026-08-26.json).

## Decision

Exceptional transport is not dead, but its supported role is now precise:

1. It is learnable and necessary on declared exceptional-action tasks.
2. The repaired E6 bundle can modestly improve the local identity bundle.
3. It does not approach Mamba-2 on generic text at this scale.
4. Dense exceptional action every token is not the production direction.
5. The next credible model is sparse/event-only exceptional transport inside a
   memory law whose addressing and transaction timing already work.

## V1.3.2 continuation

The dense-action cost conclusion above is historical and remains valid for the
v1.3.1 direct all-token chart.  Version 1.3.2 adds a different exact canonical-
product chart and a fused one-event-per-32-token SM75 recurrence.  It passes a
separate cheap-action systems gate while still missing the complete-model
Mamba-2 gate.  See
[`SM75_PRIMITIVE_TRANSPORT_RESULTS.md`](SM75_PRIMITIVE_TRANSPORT_RESULTS.md).

No v1.3.2 natural-text quality conclusion is backfilled from the systems
result.  Its parameter-matched dead-action/E6/Mamba cohort is a new experiment
with its own artifacts and fail-closed summary.
