# Endpoint-only curriculum results

Date completed: 2026-08-03.

## Result

The frozen endpoint curriculum passes every preregistered gate in all ten
seeds.

| Metric | Result |
|---|---:|
| Endpoint query compiler | 10/10 exact at 1,148 labels |
| Representation trigger | 10/10 by step 800 |
| Trigger-step range | 600--800 |
| Exact compiler invariance RMS | `3.05e-16` all seeds |
| Exact compiler homomorphism RMS | `5.02e-16` all seeds |
| Original dense L16--L256 population floor | 100% |
| Untouched class-59 dense L16--L256 population floor | 100% |
| Original/class-59 L4096 population floor | 100% |
| Class-59 L16384 population floor | 100% |
| Class-59 13-point L4096--L16384 sweep | 10/10 seeds, every point 100% |
| Maximum long-sweep path drift | `3.33e-3` |

Trigger steps were `[600, 650, 600, 600, 800, 600, 800, 600, 800, 750]`.
Thus every discovered representation emerged during L4 or the first 50 L8
updates, well before L16 training. The formal seed-0 checkpoint is bitwise identical
to the earlier implementation-smoke checkpoint in every state-dict tensor and
configuration field.

## Contrast with the failed fixed-L16 gate

Fixed L16 endpoint-only training failed at chance in seeds 0, 1, and 2 and
never triggered in 2,000 updates. The curriculum changes no endpoint-label
count: both regimes use 512,000 neural labels. It changes where the information
appears. Exact sampler analysis gives mean single-position endpoint mutual
information of 2.000 bits at L1, 1.454 at L2, 0.551 at L4, 0.040 at L8, and
0.00128 at L16. At identity initialization the measured action-gradient
direction becomes correspondingly incoherent while its norm remains nonzero.

The supported mechanistic result is:

> Finite-group endpoint training has a random-walk mixing barrier near the
> identity-symmetric initialization. Short complete words supply coherent
> first-order action gradients; once an approximate representation forms,
> exact joint-manifold retraction makes long composition reliable.

This is not ordinary vanishing-gradient evidence.

## What this pass does not yet distinguish

The curriculum simultaneously introduces short-word examples and orders them
from short to long. The shuffled-mixture arm shows that exposing the identical
batch multiset in a fully mixed order is insufficient in seed 0; it does not
show that every non-monotonic blocked order fails. The fixed-L16 8,000-step arm
tests whether persistence and a larger label budget can escape the same basin.
The separate scrambled-block control isolates monotonic continuation from
clean stage separation.

Length-1 endpoints are one-step action observations. No training example
exposes an internal prefix, but the result must not be summarized as avoiding
all short-transition information.

The compiler is also separate: it still consumes 1,148 exact endpoint queries,
including 120 designed representative extensions. The next prospective
learned-manifold compiler removes those additional queries and attempts to
recover multiplication from learned action geometry plus already-consumed
curriculum endpoint labels.

## Artifacts

- `endpoint_curriculum_10seeds.json`
- `endpoint_curriculum_long_audit_10seeds.json`
- `endpoint_credit_assignment_audit.json`
- `ENDPOINT_MIXING_BARRIER.md`
- `ENDPOINT_OPTIMIZATION_CAUSAL_PREREGISTRATION.md`
