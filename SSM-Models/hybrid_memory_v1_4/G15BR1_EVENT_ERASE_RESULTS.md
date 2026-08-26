# G15B-R1 event-anchored erase results

**Executed:** 2026-08-26 | **Start commit:**
`dba3f9a83e1ffb72555a6f9f927e6ac4c261a044` | **Start status:** clean |
**Runtime:** WSL2, RTX 2070 SUPER, SM75, PyTorch 2.9.0+cu128

**Protocol:**
[`G15BR1_EVENT_ERASE_PROTOCOL_2026-08-26.md`](G15BR1_EVENT_ERASE_PROTOCOL_2026-08-26.md)

**Artifact:**
[`g15br1_event_erase_sm75_2026-08-26.json`](artifacts/g15br1_event_erase_sm75_2026-08-26.json)

**Artifact SHA-256:**
`c015b128846e4b5c63d927778815a87728a7d613369163b1027ed3dd9f0b2912`

## Bottom line

Do not train the event-anchored erase controller.

Preserving the retained checkpoints' learned write continuation does not rescue
erase-at-every-write. Both the soft and unit event-erase arms reduce overwrite
accuracy by 9.7--11.5 points instead of improving it. They also lose 2.5--3.3
points on MQAR and 5.4--5.9 points on selective copy. All nine non-needle gates
fail; all three single-key needle cells remain perfect.

The learned address geometry is strongly nonorthogonal: mean absolute
off-diagonal live-key cosine is `0.54822` across seeds and the maximum is
`0.999934`. This makes collateral damage from the symmetric
`I - e k k^T` erase operator plausible, but does not prove it is the sole
cause. The retained code is also a distributed write program rather than an
orthogonal one-shot slot.

## Integrity

The result passes every evidentiary contract:

- the G15B parent hash is
  `f74d860e30ab40ec747521dfcecd74aac2bb75151206c25b7104d334727429eb`;
- the G15B-R0 parent hash is
  `4d92d6af2fb062cf2baaa035c4e4eff89d494dfcb56b9b666523bbbdbfe3cf9c`;
- both parents are clean-start, evidentiary quality artifacts from SM75 with
  the frozen seeds 2309, 2311, and 2333;
- all 36 retained identity cells replay query accuracy, exact-episode accuracy,
  and bits/query with maximum absolute residual `0.0`;
- the ordinary model-forward logits and reconstructed learned-control logits
  are bit-identical in every batch;
- learned query, key, value, write, retention, and transport controls are
  bit-identical across both interventions in all `3 x 1,056` evaluation
  batches;
- the local write-event decoder and collision-unobservability witness pass;
- the run performs zero optimizer updates and takes 1,726.8 seconds.

## Three-seed paired result

Mean query accuracy:

| Task and length | Learned | Soft event erase | Delta | Unit event erase | Delta |
|---|---:|---:|---:|---:|---:|
| MQAR 128 | **0.9724** | 0.9391 | -0.0333 | 0.9391 | -0.0333 |
| MQAR 512 | **0.9729** | 0.9456 | -0.0273 | 0.9456 | -0.0273 |
| MQAR 1,024 | **0.9718** | 0.9468 | -0.0251 | 0.9468 | -0.0251 |
| overwrite 128 | **0.7678** | 0.6704 | -0.0974 | 0.6704 | -0.0974 |
| overwrite 512 | **0.8328** | 0.7181 | -0.1147 | 0.7180 | -0.1147 |
| overwrite 1,024 | **0.8288** | 0.7174 | -0.1113 | 0.7174 | -0.1113 |
| selective 128 | **0.9719** | 0.9181 | -0.0539 | 0.9179 | -0.0540 |
| selective 512 | **0.9770** | 0.9229 | -0.0541 | 0.9228 | -0.0542 |
| selective 1,024 | **0.9752** | 0.9158 | -0.0594 | 0.9158 | -0.0594 |
| needle 128 | **1.0000** | **1.0000** | 0.0000 | **1.0000** | 0.0000 |
| needle 512 | **1.0000** | **1.0000** | 0.0000 | **1.0000** | 0.0000 |
| needle 1,024 | **1.0000** | **1.0000** | 0.0000 | **1.0000** | 0.0000 |

Soft and unit erase differ by at most `0.000163` in any three-seed mean cell.
That is expected from R0's retained-controller audit: learned write probability
at the labelled value token is at least `0.99935` on average in every
seed/head. Erase amplitude is therefore not the central problem.

Prototype cross-cosine by seed:

| Seed | Mean absolute off-diagonal cosine | Maximum |
|---:|---:|---:|
| 2309 | 0.38671 | 0.991306 |
| 2311 | 0.62273 | 0.999895 |
| 2333 | 0.63523 | 0.999934 |

The aggregate cross-cosine is diagnostic, not causal attribution. Seed 2309
has the lowest mean overlap but does not have uniformly least task damage.

## What failed

At every valid write, R1 applies

\[
M_t=(I-e_tk_tk_t^\top)r_tM_{t-1}+k_t(w_tv_t)^\top
\]

while preserving the learned write tail. This still assumes that `k_t` selects
an isolated replaceable slot. The evidence rejects that assumption for these
checkpoints:

1. multiple learned live-key directions are highly aligned, so symmetric erase
   can suppress unrelated associations;
2. first-write erasure is not guaranteed harmless in a shared nonorthogonal
   state;
3. the old association was written by a multi-token program, so erasing only
   the current value-token direction need not remove the stale logical value;
4. the retained decoder was trained around additive dynamics and need not
   interpret a post-hoc subtractive intervention correctly.

These mechanisms are compatible with the result; R1 does not identify one as
the unique cause.

## Narrow next checkpoint diagnostic

One factorial cell remains missing. G15B-R0 used true collision timing but
also truncated the learned write continuation. G15B-R1 preserves the write
continuation but erases on first writes and overwrites alike.

The next zero-update diagnostic must preserve every learned write control and
apply soft or unit erase only at true overwrite events. It must stratify
overwrite queries into pre-overwrite, post-overwrite same-key, and post-
overwrite other-key groups. This oracle timing is causally available from task
history but remains unobservable to the present token-local controller.

- If collision-only erase improves post-overwrite same-key recall without
  damaging the other strata, first-write collateral was decisive; a deployable
  successor still needs explicit causal occupancy state.
- If it damages post-overwrite or unrelated-key recall, symmetric learned-key
  erase is misaligned. The next mechanism test should use a dual/separate erase
  address or exact logical-component replacement, not fresh scalar-gate
  training.

## Claim boundary

This is a completed exact-SM75, three-checkpoint, zero-update causal diagnostic
on replayed G15B held-out schedules. It rejects event-at-every-write symmetric
erase for the retained learned representation. It does not falsify last-write-
wins memory generally; prove prototype cross-talk is the sole cause; validate
a state-aware, oblique, dual-address, or component-level correction; or promote
G15C, external-loss-only learning, any optimizer, Spin transport, natural
language, scaling, or a model family.
