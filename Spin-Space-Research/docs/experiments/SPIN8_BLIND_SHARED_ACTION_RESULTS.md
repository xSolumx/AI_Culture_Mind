# Spin(8) blind shared-action gate results

Date executed: 2026-08-03. Baseline commit: 315db25.
Protocol:
[SPIN8_BLIND_SHARED_ACTION_PREREGISTRATION.md](SPIN8_BLIND_SHARED_ACTION_PREREGISTRATION.md).
Raw artifact:
[spin8_blind_shared_action_seeds0_9.json](../../artifacts/spin8_blind_shared_action_seeds0_9.json).

## Frozen-gate outcome

The preregistered reliability and strong-support gates both pass.

| Criterion | Result |
|---|---:|
| joint shared-action passes | 10/10 |
| joint beats independent Lie on unseen negative action | 10/10 |
| joint beats all controls at length 2048 | 10/10 |
| unconstrained and independent-Lie controls fit visible endpoints | 10/10 |
| strong-support verdict | pass |

Every seed used a different hidden four-token action family. No seed required
resampling for commutator separation or Jacobian rank.

## Prospective observation-mask correction

The original preregistration proposed observing columns e0--e3 in the vector
and positive-chiral representations. The first pre-training rank smoke showed
that this mask has structural tangent rank 25 rather than 28; resampling could
never satisfy the validity gate. Before any accuracy result, the mask was
changed to e0--e4. It has rank 28 in every token and seed, with maximum
condition number 2.425, while still hiding:

- three columns of both visible representations;
- the complete negative-chiral action;
- every teacher matrix and bivector coefficient.

The preregistration retains this correction and its timing.

## Aggregate extrema

| Family | Maximum visible MSE | Worst one-step cosine | Worst L2048 cosine | Maximum triality error | Maximum L2048 log-norm drift |
|---|---:|---:|---:|---:|---:|
| unconstrained | 3.16e-25 | 0.8650 | -0.0506 | 0.7247 | 81.54 |
| independent polar | 3.75e-3 | 0.8651 | -0.0755 | 0.6213 | 1.68e-12 |
| independent Lie | 7.57e-14 | 0.8696 | -0.0620 | 0.8424 | 3.74e-13 |
| joint shared | 6.31e-14 | 0.99999999999935 | 0.9999999950 | 1.67e-15 | 4.59e-13 |
| oracle | 0 | 1.0 | 1.0 | 1.33e-15 | 4.46e-13 |

The independent-Lie row is the decisive control. It is exactly orthogonal and
fits the visible endpoints as tightly as the joint row, but each token and
representation has a separate tangent. It therefore retains incompatible
three-dimensional completion slack and cannot infer the entirely unobserved
negative representation.

The unconstrained row fits the visible data essentially exactly but accumulates
up to 81.5 natural-log units of norm drift by length 2048. Polar projection
fixes norm stability but not the missing shared geometry.

## Hidden-law recovery

The joint compiler consumes only the fitted visible columns. For evaluation
only, its recovered tangent coordinates were compared with the hidden teacher:
absolute coordinate cosine lies between
0.99999999999684 and 0.99999999999998 across the ten seeds.

The weakest joint metrics across the complete cohort are:

- one-step mean cosine: 0.99999999999935;
- length-2048 mean cosine: 0.9999999950;
- vector commutator ratio to oracle: 0.999999823;
- triality-equivariance error: 1.67e-15;
- parallel/recurrent error: 1.29e-14.

Thus the retraction does not merely produce some stable orthogonal completion.
It reconstructs the hidden shared action family to near machine precision and
predicts the unobserved chiral action.

## Mechanistic interpretation

Five columns of one 8D orthogonal representation leave a three-dimensional
stabilizer. Independent completions may fit every observed coordinate while
choosing incompatible points along that slack. The vector and positive
representations, tied to one 28-coordinate Spin(8) tangent, jointly remove the
stabilizer. Triality then determines the negative action without negative
labels.

This is a clean example of architectural symmetry converting partial endpoint
data into an identifiable law.

## What this does and does not establish

Established:

- blind completion of a shared Spin(8) action from incomplete endpoints;
- reliable recovery across ten independently sampled action families;
- exact cross-representation transfer to a wholly unlabelled representation;
- stable composition through length 2048;
- preservation of noncommutativity and associative scan equivalence.

Not established:

- discovery of the Spin(8) generator algebra or triality tensor;
- learned content addressing;
- language-model utility;
- superiority over delta-rule memory on retrieval;
- cross-token group-relation discovery.

The last point matters: once fixed per-token operators are identified, unseen
words compose by architectural construction. The result concerns recovery of
the correct per-token shared operators, not learning a grammar from held-out
bigram statistics.

## Next gate

Use the recovered shared-action compiler inside the dynamic-slot recurrence,
then remove supplied addressing in three stages:

1. supplied address, learned query;
2. learned token-dependent address, supplied action;
3. learned address and blind shared action.

Compare against same-width direct slots, independent-Lie transport,
Gated-Delta-style memory, linear fast weights, Householder transport, and
diagonal complex recurrence under a dense length sweep.
