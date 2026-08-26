# G15B-R2 collision-only erase results

**Executed:** 2026-08-26 | **Start commit:**
`5eae9635f010ed6411869e53bbea8d20f08cdc27` | **Start status:** clean |
**Runtime:** WSL2, RTX 2070 SUPER, SM75, PyTorch 2.9.0+cu128

**Protocol:**
[`G15BR2_COLLISION_ERASE_PROTOCOL_2026-08-26.md`](G15BR2_COLLISION_ERASE_PROTOCOL_2026-08-26.md)

**Artifact:**
[`g15br2_collision_erase_sm75_2026-08-26.json`](artifacts/g15br2_collision_erase_sm75_2026-08-26.json)

**Artifact SHA-256:**
`90652fe7034e5901b968eb5d139f02eb8bc714b0417c0889e16a2fdd6b7cf924`

## Bottom line

Do not train a scalar symmetric erase controller. Test exact logical-component
replacement next.

Perfect causal collision timing does not rescue the retained learned state
law. Restricting erase to true overwrites improves MQAR by 0.8--1.2 points and
queries before any overwrite by 1.8--2.1 points. But it makes queries after an
overwrite of the same key 10.3--12.1 points worse and lowers aggregate
overwrite accuracy by 8.7--10.4 points. Both soft and unit erase fail every
overwrite gate.

This closes the R0/R1/R2 timing/write-tail factorial. The problem is deeper
than an unobservable collision label or first-write collateral. For the
retained checkpoints, `I - e k k^T` does not remove the stale logical
association in a way the learned multi-token representation and decoder can
use.

## Integrity

Every evidentiary check passes:

- the frozen G15B, R0, and R1 parent hashes match exactly;
- all parents are clean-start evidentiary quality artifacts from SM75 with
  seeds 2309, 2311, and 2333;
- all 36 baseline cells replay query accuracy, exact-episode accuracy, and
  bits/query with maximum absolute residual `0.0`;
- ordinary model-forward and reconstructed learned-control logits are bit-
  identical in every batch;
- learned query, key, value, write, retention, and transport controls are
  bit-identical across both interventions in all `3 x 1,056` batches;
- local write-event and exact causal collision-mask checks pass in every batch;
- the R0 temporal-observability witness passes;
- the run performs zero optimizer updates and takes 1,690.2 seconds.

## Three-seed paired result

Mean query accuracy:

| Task and length | Learned | Soft collision erase | Delta | Unit collision erase | Delta |
|---|---:|---:|---:|---:|---:|
| MQAR 128 | 0.9724 | **0.9803** | +0.0079 | **0.9803** | +0.0079 |
| MQAR 512 | 0.9729 | **0.9834** | +0.0105 | **0.9834** | +0.0105 |
| MQAR 1,024 | 0.9718 | **0.9836** | +0.0118 | **0.9836** | +0.0118 |
| overwrite 128 | **0.7678** | 0.6806 | -0.0872 | 0.6804 | -0.0874 |
| overwrite 512 | **0.8328** | 0.7287 | -0.1041 | 0.7287 | -0.1041 |
| overwrite 1,024 | **0.8288** | 0.7289 | -0.0999 | 0.7290 | -0.0998 |
| selective 128 | **0.9719** | 0.9608 | -0.0111 | 0.9608 | -0.0111 |
| selective 512 | **0.9770** | 0.9647 | -0.0123 | 0.9647 | -0.0123 |
| selective 1,024 | **0.9752** | 0.9657 | -0.0095 | 0.9657 | -0.0095 |
| needle 128 | **1.0000** | **1.0000** | 0.0000 | **1.0000** | 0.0000 |
| needle 512 | **1.0000** | **1.0000** | 0.0000 | **1.0000** | 0.0000 |
| needle 1,024 | **1.0000** | **1.0000** | 0.0000 | **1.0000** | 0.0000 |

The soft and unit arms differ by at most `0.000163`, again confirming that
erase amplitude is not the limiting variable at the saturated write event.

## Causal overwrite strata

Each seed/length has 512 pre-overwrite and 3,584 post-same-key decisions. The
commissioned overwrite schedule contains no `after_unrelated_overwrite_only`
queries, so R2 provides no evidence for that registered guard stratum.

Soft collision erase:

| Length | Before overwrite learned | Repaired | Delta | After same-key overwrite learned | Repaired | Delta |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.8639 | **0.8848** | +0.0208 | **0.7541** | 0.6514 | -0.1027 |
| 512 | 0.9329 | **0.9505** | +0.0176 | **0.8185** | 0.6970 | -0.1215 |
| 1,024 | 0.9245 | **0.9440** | +0.0195 | **0.8151** | 0.6982 | -0.1169 |

Unit collision erase is numerically the same to four decimals. The direction
of the effect is unambiguous: removing learned erase outside collisions is
helpful, but applying the symmetric contraction exactly where replacement is
needed is harmful.

## What this establishes

R2 rules out two narrower explanations for the retained-checkpoint failure:

1. **It is not mainly first-write collateral.** No intervention erase is
   applied before a true collision, yet post-overwrite accuracy still falls.
2. **It is not mainly erase amplitude.** Soft and unit arms are effectively
   identical because the learned write event is saturated.

The remaining supported diagnosis is representation/update-law mismatch. A
logical association is written by a distributed multi-token program into a
shared, nonorthogonal fast-weight state. A symmetric contraction along the
current value-token key removes some state mass, but not the complete stale
association component, and can change features the fixed decoder expects.

R2 does not isolate which of key drift, the write tail, channelwise retention,
learned background injection, or decoder adaptation is individually
responsible. It shows that perfect collision timing is insufficient for this
operator and checkpoint representation.

## Next mechanism: exact logical-component replacement

The next zero-update oracle should decompose the linear state into a background
component and one component per logical key:

\[
M_t=C_t^{(0)}+\sum_j C_t^{(j)}.
\]

Every component receives the same learned linear transition. The complete
value-token plus one-token-tail injection is assigned to its logical key. At
an overwrite of key `j`, reset `C^{(j)}` before adding the new learned write
program; leave every other component untouched. This exactly tests last-write-
wins at the representation actually learned, without assuming orthogonal keys
or a rank-one erase direction.

The construction remains affine-scan compatible: each key component is a
parallel scan whose left transition is zeroed at that key's overwrite events.
It is an oracle slot decomposition and expands state by the number of live
keys. A pass would motivate a learned slot/occupancy architecture with explicit
replacement. A failure would show that these checkpoints' decoder relies on
additive history strongly enough that a new model must be trained around the
correct state law rather than repaired post hoc.

## Claim boundary

This is a completed exact-SM75, three-checkpoint, zero-update causal diagnostic
on replayed G15B schedules. It rejects collision-only symmetric learned-key
erase for the retained representation. It does not falsify delta rules,
GDN2/KDA, separate or dual erase addresses, explicit slot memories, or last-
write-wins learning generally. It does not provide evidence for fresh training,
G15C, the current token-local controller, an optimizer, Spin transport,
ordinary text, scaling, or a model family.
