# Joint blind Spin(8) action and continuous-alias results

Date executed: 2026-08-03. Prior frozen baselines: `8b86e7a`, `4eb7459`,
and `fede409`.

Protocol:
[SPIN8_BLIND_ALIAS_ACTION_PREREGISTRATION.md](SPIN8_BLIND_ALIAS_ACTION_PREREGISTRATION.md).
Raw artifact:
[spin8_blind_alias_action_seeds0_9.json](../../artifacts/spin8_blind_alias_action_seeds0_9.json).
SHA-256: `377bf8fe47516610f615610992743f36b3521fe6ab8c8e423f606243c85b4463`.

## Verdict

The combined gate passes every frozen criterion. In 10/10 independently
sampled worlds, one optimization run jointly learns:

- a collision-free write/query router from fresh continuous aliases, without a
  logical key ID; and
- one shared 28-coordinate Spin(8) action family whose vector and
  positive-spinor views complete the negative-spinor action outside the
  supplied rank-2 calibration plane.

The learned recurrence remains exact through the dense length sweep from 16 to
2048 with a 64-scalar streaming state. The decisive independent control has
three times as many action coordinates and fits every supplied endpoint to
near machine precision, but its unobserved negative action is wrong and its
direct long-horizon retrieval fails.

This is a relational-completion result. It does not show that triality is a
better generic memory than direct slots: when the correct negative action is
supplied, the direct-slot oracle is also exact.

## Frozen-gate outcome

| Criterion | Result |
|---|---:|
| joint triality binding passes | 10/10 |
| matched joint-direct passes | 10/10 |
| exact-action + learned-alias decomposition passes | 10/10 |
| learned-action + oracle-alias decomposition passes | 10/10 |
| independent controls fit supplied evidence | 10/10 |
| shared negative-complement completion wins | 10/10 |
| shared versus independent-direct L2048 wins | 10/10 |
| independent-binding behavioral bypasses | 10/10 |
| complete frozen gate | pass |

Every hidden four-token teacher passed the prospective commutator and rank
contracts without resampling.

## Identifiability audit

The Jacobians reproduce the preregistered structure in every seed:

| Family/view | Tangent coordinates | Observed rank |
|---|---:|---:|
| shared Spin(8) family | 28 | 28 |
| independent vector action | 28 | 25 |
| independent positive-spinor action | 28 | 25 |
| independent negative rank-2 endpoints | 28 | 13 |
| independent total | 84 | 63 |

Thus the independent family retains 21 unobserved tangent dimensions. The
shared family has no local tangent slack under the same visible evidence. This
rank statement is also covered by a separate test seed in
`test_foundational_contracts.py`.

The shared coordinate recovery is not merely functionally adequate: the worst
cosine between the learned and hidden 28-coordinate token families is
`0.9999999999999953`.

## Aggregate extrema

| Metric | Joint direct | Independent direct |
|---|---:|---:|
| maximum supplied-column MSE | 7.98e-17 | 1.68e-16 |
| maximum final training loss | 1.80e-15 | 2.14e-14 |
| worst held-out negative-complement cosine | 0.9999999999999988 | 0.863755 |
| best independent negative-complement cosine | -- | 0.929780 |
| worst dense mean retrieval cosine | 0.9999999999999839 | 0.407763 (direct) |
| worst individual retrieval cosine | 0.9999999999995393 | -0.977690 (direct) |
| maximum dense mean relative squared error | 3.26e-14 | 1.18447 (direct) |
| maximum triality-equivariance error | 1.55e-15 | 0.721221 |
| minimum commutator ratio to oracle | 0.999999959 | 0.999999681 |
| maximum prefix-scan/recurrent error | 6.67e-16 | 6.67e-16 |

All test write and query aliases have 100% agreement with their untouched
class centers in all variants. The largest independent training terms are:

- supplied action observation MSE: `1.68e-16`;
- rank-2 binding endpoint MSE: `1.92e-14`;
- rank-2 direct endpoint MSE: `6.98e-16`.

The independent failure is therefore not failure to optimize the supplied
training problem.

## Matched-path direct behavioral separation

The ten independent-direct L2048 mean cosines are:

`0.4833, 0.5295, 0.4960, 0.5362, 0.4686, 0.5212, 0.5021, 0.4563, 0.4078, 0.5458`.

Every corresponding shared value is equal to 1.0 at the displayed precision.
Both families are orthogonal, retain nontrivial commutators, use the same
aliases, receive the same endpoint batches, and fit the supplied evidence.
The difference is the shared representation constraint: one tangent element
must generate all three triality actions.

The shared-direct row was added after the first cohort exposed a path mismatch:
the original shared row used binding while the independent comparison used
direct transport. That first artifact was invalidated, the correction was
recorded in the protocol, and all ten seeds were rerun from the top without
changing training or thresholds. The original learned coordinates and metrics
reproduced exactly. Shared-direct and shared-binding both have worst dense mean
cosine `0.9999999999999839`; only the direct row is used for the causal
negative-action comparison.

## The binding bypass is a mechanistic warning

`independent_binding` also retrieves nearly exactly through length 2048, yet it
passes zero mechanism gates. This is not contradictory. Its inference path
transports a bound vector with the learned vector action, transports the query
key with the learned positive-spinor action, and unbinds. It never consumes its
learned negative-spinor action. Consequently it can solve the behavioral task
while its negative complement cosine is only 0.864--0.930 and its triality
residual reaches 0.721.

The seed-0 smoke exposed the independent binding bypass before seeds 1--9 were
run. A later post-cohort source-flow audit caught that the shared behavioral row
also used binding. The protocol records both corrections and their timing. A
regression test now changes only the negative action and proves that binding
metrics remain exactly invariant while direct metrics degrade.

This separation is itself useful: retrieval accuracy alone cannot certify
that a purported multiview action family is geometrically coherent.

## What is established

- Joint family retraction and joint alias routing can be optimized together in
  one endpoint-supervised run.
- The shared 28-coordinate family completes an action on an entirely held-out
  six-dimensional complement where a parameter-richer independent family is
  locally underdetermined.
- The completion remains accurate under noncommuting compositions through
  length 2048 in both binding and matched direct recurrence paths.
- The address encoder remains outside the recurrent state update, so the
  recurrence is affine and parallel/recurrent equivalent.
- The result is reliable across ten hidden action families and ten alias
  worlds under one frozen optimizer schedule.

## What is not established

- The Spin(8) generators, triality tensor, eight-class ontology, and latent
  slot count are architectural priors.
- Five vector and five positive-spinor matrix columns per action token are
  supplied during training; the action law is incomplete, not label-free.
- Alias classes are balanced perturbations of orthogonal hidden centers.
- The rank-2 calibration plane is synthetic and known only through sampled
  endpoints.
- The result does not beat a direct memory supplied with the correct negative
  transport, nor full Gated DeltaNet-2, EDA, Q-Delta, or linear-attention
  systems on naturalistic tasks.
- It does not yet establish language-model utility or throughput advantage.

## Next decisive gate

The next experiment should remove the remaining matrix-column oracle without
changing the recurrence:

1. replace supplied vector/positive action columns with paired transformed
   examples generated from continuous observations;
2. learn the shared action, alias router, and decoder from sequence endpoints;
3. sweep observation rank, calibration rank, noise, class imbalance, and
   `K/H` rather than evaluating one favorable point;
4. compare against an equal-state direct latent operator compiler, delta-rule
   memory with learned erasure, Householder transport, and a generic
   equivariant multilinear control;
5. retain the rank audit, untouched aliases, dense length sweep, and
   mechanistic/behavioral split as mandatory gates.

Only a win under that weaker supervision would justify moving from
"architectural relational completion" toward "discovery of useful geometric
transport."
