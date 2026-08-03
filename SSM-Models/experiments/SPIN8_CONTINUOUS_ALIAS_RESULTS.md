# Spin(8) continuous-alias address results

Date: 2026-08-03.

Protocol: `SPIN8_CONTINUOUS_ALIAS_PREREGISTRATION.md`.
Raw artifact: `spin8_continuous_alias_seeds0_9.json`.
SHA-256: `56e3eda9f542849bbfec9012ce14b3991940f86a0597532d895bc78d6b17dec0`.

## Verdict

The logical key ID has been removed in this controlled setting. Separate
write and query encoders receive only fresh 24-dimensional continuous aliases.
Jointly balanced slot routing passes 10/10 seeds and remains exact over unseen
mixed-key write/overwrite/rotate/query streams at every dense length from 16
through 2048. Test aliases use radius 0.35, beyond the largest training radius
0.15, and are never repeated.

The result is content-derived routing for an orthogonal latent-cluster model.
It is not natural-language semantics, and it is not a Spin(8)-specific win:
same-width direct slots pass with identical cosine in every cell.

## Frozen-gate table

| Variant | Passes | Worst dense mean cosine | Worst dense mean relative squared error | Causal reading |
|---|---:|---:|---:|---|
| oracle write + oracle query | 10/10 | 1.000000 | 3.66e-26 | numerical oracle |
| oracle write + learned query | 10/10 | 1.000000 | 3.66e-26 | query alias inference works |
| learned write + oracle query | 10/10 | 1.000000 | 3.66e-26 | write alias inference works |
| learned both, independent | 0/10 | 0.448503 | 1.1030 | local fit, global collisions |
| learned both, joint balance | 10/10 | 1.000000 | 3.66e-26 | complete alias gate |
| untrained learned-both | 0/10 | 0.028415 | 1.4364 | architecture alone is insufficient |
| direct slots, joint route | 10/10 | 1.000000 | 7.90e-32 | exact parity with triality |
| delta, oracle semantic projector | 10/10 | 1.000000 | 3.39e-31 | delta recurrence has capacity |
| fast weights, oracle projector | 0/10 | 0.184488 | 46.5115 | additive writes cannot overwrite |
| delta, learned keys | 0/10 | 0.800833 | 0.4086 | continuous key inference is imperfect |
| fast weights, learned keys | 0/10 | 0.153098 | 46.4878 | inference plus overwrite failures |

Every dense cell contains at least 312 query events. All principal recurrent
states contain exactly 64 scalars.

## What independent routing proves

The independent row is a particularly strong negative control:

- final single-key endpoint loss is at most `1.93e-31`;
- write/query center assignments agree for all eight classes in every seed;
- untouched write and query aliases agree with their own center assignment
  100% of the time;
- alias entropy is at most `1.46e-29`;
- nevertheless every seed has one to four class collisions.

Thus its failure is neither optimization, encoder misalignment, alias OOD
generalization, nor soft routing. It is precisely the relational constraint
missing from independently normalized families.

Joint marginal balance removes that slack without a class-specific target.
All ten joint seeds have zero write/query collisions, complete cross-encoder
agreement, 100% untouched-alias agreement, and endpoint/balance losses at
floating-point zero. The training-only constraint never enters the recurrence.

## Scan and OOD contracts

World construction passes exact audits across all seeds:

- center Gram maximum error: `6.67e-16`;
- analytic radius-cosine maximum error: `6.67e-16`;
- CPU/CUDA center mismatch: exactly zero.

Because aliases are `(c_k + r u) / sqrt(1+r^2)` with unit nuisance orthogonal
to the complete center span, test center cosine is exactly
`1/sqrt(1+0.35^2) ~= 0.943858`. The OOD change is therefore an explicit radius
increase, not a changed world or device-specific distribution.

Worst prefix-scan versus recurrence error across every family is `9.99e-16`.
The alias encoder is evaluated before the scan and never reads recurrent
state; no dynamic Sinkhorn operation occurs inside sequence execution.

## Delta and fast-weight diagnosis

The oracle projector makes delta overwrite exact in 10/10 seeds. Therefore
learned-delta failure is not a 64-scalar capacity mismatch or a faulty update.
Nine learned seeds reach length-2048 cosine around 0.972--0.981, while seed 9
falls to 0.811. Across the cohort, center Gram error reaches 0.075 and minimum
write/query center cosine falls to 0.865. Distributed keys inherit these small
angular and occasional alignment errors continuously over long streams.

Hard slots behave differently: once an alias remains on the correct side of a
decision boundary, the route is exactly the same vertex. This is evidence for
a quantization/cleanup robustness advantage of jointly balanced slots over the
tested learned delta parameterization under equal state width and training
budget. It is not a general result against Gated DeltaNet or every possible
delta-key encoder.

Fast weights fail even with exact oracle keys: length-2048 cosine is only
0.184--0.196 and relative error grows above 46 because additive writes retain
obsolete values. This cleanly isolates the need for erase/delta correction.

## Claim boundary and next gates

The supplied conditions remain strong: eight orthogonal centers, eight slots,
balanced class frequency, supplied noncommuting Spin(8) actions, and a
controlled nuisance model. The correct next order is:

1. jointly learn the Spin(8) action family and continuous alias router in one
   endpoint-supervised model rather than passing actions externally;
2. relax center orthogonality and class balance, then test `K>H` with explicit
   eviction/tight-frame capacity accounting;
3. move to naturalistic selective-copy and MQAR splits with full Gated
   DeltaNet, erase-then-delta, linear-attention, and fast-weight baselines;
4. claim a Spin(8) benefit only if triality beats direct slots on state,
   sample, extrapolation, or transport efficiency.
