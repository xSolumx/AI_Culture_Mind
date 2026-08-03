# Spin(8) learned-address gate results

Date: 2026-08-03.

Protocol: `SPIN8_LEARNED_ADDRESS_PREREGISTRATION.md`.
Raw artifact: `spin8_learned_address_seeds0_9.json`.
SHA-256: `8305bc78907cec2b20859f14b8497b5d9bea661dee25dd54f9db1db1737bc5ba`.

## Verdict

The joint latent-address gate passes 10/10 seeds.  Training sees only
single-key episodes.  Nevertheless, every jointly normalized router becomes
a collision-free permutation and retrieves exactly on previously unseen
mixed-key write/overwrite/query streams through length 2048.

This is a real structural result, but it is not a triality advantage.  The
same-width direct-slot control also passes 10/10.  The result isolates joint
family normalization as the source of global address consistency; triality
supplies equivariant binding and vector-representation transport, not superior
capacity on this task.

## Frozen-gate table

| Variant | Passes | Rounded collisions across seeds | Worst dense mean cosine | Worst dense mean relative squared error |
|---|---:|---:|---:|---:|
| triality oracle address | 10/10 | all 0 | 1.000000 | 3.78e-26 |
| untrained joint Sinkhorn | 0/10 | 0--3 | 0.422425 | 0.8345 |
| trained independent rows | 0/10 | 1--4 | 0.404981 | 1.1900 |
| trained joint Sinkhorn | 10/10 | all 0 | 1.000000 | 3.78e-26 |
| same-width direct joint | 10/10 | all 0 | 1.000000 | 8.07e-32 |

Every reported dense cell contains at least 328 query events.  Lengths are
16, 32, 64, 128, 256, 512, 1024, and 2048.

## Mechanistic separation

The independent router is not an optimization failure.  Its single-key
training loss reaches `1.97e-18` to `5.53e-18`, and its rows become effectively
one-hot (mean entropy below `5.2e-38`).  It fails only when independently
perfect rows choose the same latent slot: every seed has one to four
collisions, and length-2048 mean cosine ranges from 0.507 to 0.879.

The untrained joint control also fails 0/10.  Its mean row entropy remains
1.380--1.734 and its length-2048 cosine ranges from 0.422 to 0.619.  Seed 6
even rounds to a collision-free assignment but remains too soft to retrieve.
Thus the pass is not supplied by running a random matrix through Sinkhorn:
endpoint optimization is required to reach the vertices.

The trained joint family combines both ingredients:

- endpoint loss drives each row to a simplex vertex;
- double stochasticity makes distinct rows occupy distinct columns.

All ten learned matrices have zero measured row/column residual at float64
evaluation temperature and mean row entropy between `2.1e-38` and `1.7e-27`.
The accompanying `SPIN8_LATENT_ADDRESS_THEOREM.md` proves why this intersection
contains only permutations when keys equal slots.

## Recurrent and geometric contracts

- streaming state: 64 scalars (`8 slots x 8 coordinates`), independent of
  sequence length;
- worst parallel-prefix versus recurrent error: `5.55e-16`;
- worst rotation-only absolute log-norm drift through 2048 actions:
  `6.87e-13`;
- minimum individual query cosine for both passing learned rows:
  `0.9999999999999996`;
- joint beats independently normalized routing in collision count and
  length-2048 cosine in 10/10 paired seeds;
- trained joint beats its untrained initialization at length 2048 in 10/10.

The recurrence remains affine because routing depends on the input key, never
the running state.  The logarithmic-depth scan and constant-state streaming
implement the same transition family.

## What changed scientifically

The blind-action experiment showed that independently correct orthogonal
actions can retain unobserved representation slack, while joint retraction
removes it.  This experiment finds the discrete analogue: independently
perfect address rows retain collision slack, while joint retraction onto the
Birkhoff polytope removes it.

The shared principle is now supported in both continuous and discrete forms:

> Optimize local tangent or address variables freely, but retract the entire
> token family onto one shared representation manifold.  Never normalize
> tokens independently when the missing constraint is relational.

## Claim boundary and next falsifier

The logical key ID is still supplied.  Any learned permutation is equally
valid because slots are latent.  Therefore this result does not show semantic
address inference from raw content.

The next gate should replace the logical-ID oracle with multiple continuous
aliases per semantic key.  Training should pair write aliases with disjoint
query aliases, reserve unseen aliases and all mixed-key episodes, and compare:

1. supplied address plus learned query;
2. learned address plus supplied action;
3. learned address and learned query;
4. same-width direct memory;
5. delta-rule, erase-then-delta, bilinear fast-weight, and linear-attention
   controls.

Only an advantage in state efficiency, alias extrapolation, sample efficiency,
or transport structure would support a Spin(8)-specific claim.
