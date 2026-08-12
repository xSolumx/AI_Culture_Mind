# Endpoint-only fixed-length result

Date closed: 2026-08-03.

## Outcome

The preregistered ten-seed gate failed at seed 0, so it was not meaningful to
spend the full cohort merely to obtain a larger pass-count denominator. Seeds
1 and 2 were run unchanged as a diagnostic replication. All three runs failed
in the same way.

| Seed | Compiler labels | States covered | Edges observed/completed | Trigger | Minimum logged task loss | Final task loss | Maximum logged endpoint accuracy | Final accuracy |
|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 0 | 1,148 | 60/60 | 120/120 | no | 4.07 | 4.08 | 5% | 2% |
| 1 | 1,148 | 60/60 | 120/120 | no | 4.07 | 4.09 | 4% | 4% |
| 2 | 1,148 | 60/60 | 120/120 | no | 4.06 | 4.08 | 5% | 2% |

The chance cross-entropy is `log(60) = 4.0943` and chance accuracy is 1.67%.
No seed approached the representation-discovery trigger during 2,000 updates.

## What failed and what did not

The endpoint compiler did not fail. Every run covered all 60 anonymous labels,
inferred the exact inverse matching, observed 120 representative-extension
edges, completed the other 120, and recovered the regular action. The failed
component was neural optimization from one terminal label on fixed length-16
words.

This is a **0/1 failure of the preregistered all-seed gate**, accompanied by a
3/3 diagnostic replication of the chance plateau. It is not reported as a
formal 0/10 cohort because seven seeds were deliberately not run after the
gate was already logically lost.

## Mechanistic diagnosis

The prospective credit-assignment audit separates gradient disappearance from
gradient cancellation. Under the exact training sampler, mutual information
between one token position and the final group element falls from 2.000 bits
at length 1 to a mean 0.00128 bits at length 16 (position range
0.000113--0.00593 bits). At identity initialization, the empirical
action-gradient cosine to its 32-batch mean falls from 0.994 at length 1 to
0.268 at length 16, even though the RMS batch-gradient norm remains nonzero
(0.381 at length 16).

The supported explanation is therefore first-order signal cancellation after
the generator random walk mixes over A5, not ordinary vanishing gradients.
This is still a diagnosis of the initialization boundary, not a proof that no
fixed-length optimizer or larger label budget could ever escape it.

## Next controlled gate

The separately preregistered endpoint curriculum remains unchanged. A pass
would show that short complete words can bootstrap the endpoint-only neural
optimizer. It would not show that passive length-16 endpoints suffice, nor
that the schedule avoids one-step action information: length-1 endpoints are
one-step observations even though no example reveals an internal prefix.

The causal follow-up compares the ordered curriculum with the identical
short-word multiset in shuffled order and with a longer fixed-L16 persistence
control. Those arms distinguish staged continuation, short-word exposure, and
larger endpoint-label budget.
