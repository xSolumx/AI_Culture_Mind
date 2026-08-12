# Spin(8) continuous-alias address gate preregistration

Date frozen: 2026-08-03, before implementation or results.
Baseline commit: `4eb7459`.

## Pre-reliability implementation correction

A seed-0, 50-step-per-stage smoke failed every learned row despite essentially
zero final training endpoint loss. The cause was traced before any reliability
run: constructing the seeded semantic basis separately with CPU and CUDA RNGs
produced different worlds at train and evaluation time. World construction is
now canonicalized in float64 on CPU and then copied to the requested device.
The failed smoke is an implementation diagnostic, not part of the cohort; no
threshold or model hyperparameter changed in response.

## Question

Can a state-independent router infer latent key identity from previously unseen
continuous aliases, align disjoint write/query alias domains, and retain exact
mixed-key memory behavior without ever receiving a logical key ID?

## Data and oracle boundary

Each seed samples eight orthogonal latent semantic centers in 24 dimensions.
An observed alias is the normalized sum of its center and fresh nuisance drawn
from the orthogonal complement. The model receives only the 24-vector alias.
It never receives the center, latent class index, target slot, or an oracle
nearest-center score.

Training batches contain balanced but independently ordered single-key
episodes. Each episode pairs a fresh write alias with a fresh query alias for
the same hidden semantic key and supplies only the retrieved value endpoint.
No training recurrence contains two semantic keys. Balance regularizers may
use the unlabeled marginal address/key distribution over a balanced batch;
they may not use class-indexed slot targets or cross-key retrieval labels.

Alias domains are disjoint by RNG stream and noise radius:

- curriculum train radii: 0.05, 0.10, and 0.15;
- validation radius: 0.22;
- untouched test radius: 0.35.

Because nuisance is unit-normalized and exactly orthogonal to the full center
span, every radius-`r` alias has analytic center cosine
`1 / sqrt(1 + r^2)`. This invariant, the center Gram matrix, and cross-device
world identity must be tested directly; the OOD shift is therefore a specified
support-radius change, not an accidental RNG-density change.

All mixed-key sequences and all validation/test alias samples are prospective.
Validation may report diagnostics but may not select per-seed hyperparameters.

## Address variants

1. `oracle_both`: supplied write and query addresses;
2. `oracle_write_learned_query`: write address supplied, query inferred;
3. `learned_write_oracle_query`: write inferred, query supplied;
4. `learned_both_independent`: both inferred with local endpoint and vertex
   losses only;
5. `learned_both_joint`: both inferred with the same local losses plus a
   shared marginal-Birkhoff constraint;
6. `learned_both_joint_untrained`: identical joint architecture without
   endpoint updates;
7. `direct_joint`: the learned joint router with same-width direct slots;
8. `triality_joint`: the learned joint router with triality bind/unbind and
   supplied Spin(8) transport.

Write and query encoders are separate linear maps. Their only alignment signal
is same-episode retrieval. The joint constraint sees only mean slot usage and
therefore cannot prescribe which semantic class owns which latent slot.

The endpoint is already a cross-encoder constraint. For simplex routes, exact
one-write retrieval requires write/query overlap one and hence the same vertex.
For normalized delta keys, it requires `q^T k = 1` and hence `q = k` by the
equality case of Cauchy--Schwarz. Independent encoder gauges cannot be hidden
inside a learned memory matrix because no such compensator exists. A common
permutation (slots) or common orthogonal gauge (delta keys) remains and is
intentionally unidentifiable.

## Matched recurrent baselines

The recurrent state budget is 64 scalars for every principal row.

- `delta_joint`: learned unit key vectors with an unlabeled covariance-
  whitening constraint and delta-rule overwrite;
- `fast_weight_joint`: the same learned keys with additive bilinear fast-
  weight writes and no erase correction;
- `direct_joint`: eight direct slots with the jointly balanced router;
- `triality_joint`: eight triality-bound vector slots.

The delta/fast-weight key encoder receives the same aliases and endpoint loss.
Whitening may use only the balanced marginal second moment, not class labels.
If direct or delta memory matches triality, no Spin(8)-specific advantage may
be claimed.

Before the ten-seed cohort produced any result, a full-budget seed-0 smoke
showed learned delta keys below the retrieval gate. Two capacity diagnostics
are therefore added without changing any gate: `delta_oracle` and
`fast_weight_oracle` use the exact semantic-center projector. They distinguish
failure to infer nuisance-invariant keys from failure of the recurrent update
itself. They are labelled oracles and cannot support a learned-model claim.

## Recurrent contract

All encoders depend only on the current alias. Routing/key generation never
reads recurrent state. Slot, delta, and fast-weight updates are affine in the
state once the current input has been encoded; parallel prefix composition and
constant-state streaming must be numerically equivalent.

No Sinkhorn iteration runs inside the recurrence. The joint slot constraint is
an unlabeled marginal loss during training; inference uses a single linear map
and softmax per alias. Encoded routes are external transition coefficients, so
backpropagation through the router does not weaken affine scan composition.

Supplied noncommuting Spin(8) action tokens transport triality memory in the
vector representation and direct/delta/fast-weight values in the negative
chiral representation. Alias features are semantic addresses and do not rotate
with the geometric value frame.

## Optimization

- seeds 0 through 9;
- fixed three-stage alias-radius curriculum;
- Adam, fixed schedule shared across paired variants;
- float64 final evaluation;
- local GPU when useful;
- no per-seed restarts, threshold tuning, or checkpoint selection.

## Evaluation

- single-key train and validation endpoint loss;
- held-out alias assignment consistency relative to each model's own latent
  training permutation/directions, never an oracle slot labeling;
- mixed-key write/overwrite/rotate/query lengths 16, 32, 64, 128, 256, 512,
  1024, and 2048;
- at least 256 query events per seed/length cell;
- rounded class collisions, row/column marginal residuals, entropy or key Gram
  residual, query cosine, relative squared error, scan parity, norm drift, and
  recurrent-state size.

Rounded collisions are diagnostic rather than a differentiable objective or a
standalone success measure. No straight-through estimator is used. Continuous
entropy, marginal/Gram residuals, alias agreement, and retrieval errors remain
visible beside the rounded count.

## Frozen gates

An alias-derived joint slot seed passes only if:

- all eight semantic centers map to distinct rounded slots;
- untouched test aliases agree with their model-derived center slot at least
  99% of the time for both write and query encoders;
- mean query cosine is at least 0.995 at every dense length;
- minimum individual query cosine is at least 0.98;
- mean relative squared retrieval error is below 1e-3 at every dense length;
- parallel/recurrent error is below 1e-9;
- streaming state is exactly 64 scalars.

Reliability requires at least 8/10 passing `triality_joint` seeds. Strong
support for the shared-family mechanism additionally requires:

- both one-sided oracle decompositions pass at least 8/10;
- independent routing fits single-key training below 1e-6 yet has more
  collisions or lower length-2048 cosine than joint routing in at least 8/10;
- trained joint beats untrained joint at length 2048 in 10/10;
- the direct control is reported under the identical learned route family.

Baseline results are comparative, not pre-required to fail. A Spin(8)-specific
claim requires triality to beat both direct slots and delta overwrite on state
efficiency, alias extrapolation, sample efficiency, or transport behavior.

## Interpretation boundary

Passing shows content-derived address inference for a controlled continuous
cluster model, not semantic understanding, raw-language routing, or general
MQAR. Orthogonal latent centers and balanced class frequency are architectural
conditions. The next tier must relax those conditions and use naturalistic
selective-copy/MQAR data.
