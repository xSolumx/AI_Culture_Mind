# G15B-R3 exact logical-component replacement results

**Completed:** 2026-08-26  
**Frozen protocol:**
[`G15BR3_LOGICAL_COMPONENT_PROTOCOL_2026-08-26.md`](G15BR3_LOGICAL_COMPONENT_PROTOCOL_2026-08-26.md)  
**Exact-SM75 artifact:**
[`artifacts/g15br3_logical_component_sm75_2026-08-26.json`](artifacts/g15br3_logical_component_sm75_2026-08-26.json)  
**Artifact SHA-256:**
`0fe54b8ce38868d67a7ecb0cb888f2279d8809c2bbaf3ccbda678326ff808959`

## Decision

**G15B-R3 fails its frozen promotion gate, but the component-replacement
mechanism is not rejected. Do not train it yet; inspect write-tail ownership
and background/component coupling.**

The oracle `erase_free_lww` arm repairs ordinary overwrite strongly. It gains
12.2--12.8 accuracy points over learned symmetric erase and 51.9--55.5 points
over the erase-free no-reset control. On post-same-key queries, it gains
13.6--14.3 points over learned erase and 59.3--63.4 points over no reset. It
also reaches exactly `1.0` accuracy in every constructed guard cell and
preserves its unrelated-key strata.

This is genuine mechanism evidence for exact component replacement, but not a
training authorization. The frozen gate also required a 10-point gain over
the learned arm on the constructed guard. That arm was already
`0.999674`--`0.999919`, so the registered improvement check fails even though
replacement is perfect. Separately, the learned component replay exceeds the
frozen FP32 logit tolerance in two seeds. Query predictions remain identical
and state residuals pass, but the preregistered integrity conjunction must be
reported as failed.

## Execution contract

- clean start commit: `3e2e5f0e7caed16b112c7c0dda38182f2aff8e72`;
- clean status at start: `[]`;
- runtime: WSL Python 3.11.16, PyTorch `2.9.0+cu128`, CUDA 12.8;
- device: NVIDIA GeForce RTX 2070 SUPER, compute capability `(7,5)`;
- retained identity checkpoints: seeds 2309, 2311, and 2333;
- tasks: MQAR, overwrite, constructed overwrite guard, selective copy, and
  needle;
- lengths: 128, 512, and 1,024;
- support: 4,096 query decisions per seed/task/length cell;
- optimizer updates: zero;
- elapsed wall time: 3,485.49 seconds.

The artifact binds and verifies G15B plus R0, R1, and R2 by their frozen
hashes. All 36 retained G15B baseline cells replay with zero accuracy,
exact-episode, bits/query, and no-erase residual.

## Three-seed mean accuracy

### Ordinary overwrite

| Length | Learned | Erase-free, no reset | Erase-free LWW | LWW - learned | LWW - no reset |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.767822 | 0.376465 | **0.895752** | **+0.127930** | **+0.519287** |
| 512 | 0.832764 | 0.413411 | **0.954346** | **+0.121582** | **+0.540934** |
| 1,024 | 0.828776 | 0.398519 | **0.953613** | **+0.124837** | **+0.555094** |

For the populated `after_same_key_overwrite` stratum:

| Length | Learned | Erase-free, no reset | Erase-free LWW | LWW - learned | LWW - no reset |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.754092 | 0.303850 | **0.897321** | **+0.143229** | **+0.593471** |
| 512 | 0.818452 | 0.336682 | **0.954892** | **+0.136440** | **+0.618211** |
| 1,024 | 0.815104 | 0.320592 | **0.954985** | **+0.139881** | **+0.634394** |

Every ordinary-overwrite aggregate and post-same-key improvement gate passes
against both controls. The original overwrite generator still contains no
`after_unrelated_overwrite_only` query; no claim for that stratum comes from
the ordinary cohort.

### Constructed guard

| Length | Learned | Erase-free, no reset | Erase-free LWW | LWW - learned | LWW - no reset |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.999756 | 0.806234 | **1.000000** | +0.000244 | +0.193766 |
| 512 | 0.999919 | 0.806966 | **1.000000** | +0.000081 | +0.193034 |
| 1,024 | 0.999674 | 0.803955 | **1.000000** | +0.000326 | +0.196045 |

Across the three seeds, every guard length contains 3,072 queries before any
overwrite, 4,608 after only an unrelated overwrite, and 4,608 after a same-key
overwrite. `erase_free_lww` reaches `1.0` in every stratum. It therefore passes
all registered preservation checks for the first two strata and the 10-point
same-key improvement against no reset. It fails the registered 10-point
aggregate and same-key improvement checks against `learned`, whose guard
accuracy is already saturated. The gate is retained exactly as frozen; this
result is not relabelled as a pass after inspection.

### Non-overwrite guards

- MQAR improves over learned by `+0.00789`, `+0.01050`, and `+0.01180`.
- Selective copy trails learned by `-0.01115`, `-0.01229`, and `-0.00952`,
  inside the frozen `-0.02` floor.
- Needle remains exactly `1.0` at all three lengths.

These gates all pass. LWW and no reset are identical on these tasks, as
expected because no same-key component replacement is requested.

## Integrity result

Baseline replay, parent hashes, the temporal-observability witness, local
write decoding, ownership masks, preserved non-erase controls, and query
predictions pass. The diagnostic also records 413 final-token writes without
an invented tail. Across seeds, owned in-range tails comprise 220,054 filler
tokens, 13,468 following write markers, and 1,073 item markers. This confirms
that the frozen `t+1` convention is not synonymous with filler ownership.

The expanded state is nine times the base recurrent state for the eight-key
tasks: 2,304 versus 256 scalars per sequence. The two-key guard uses 512 versus
256. It is neither state-, parameter-, compute-, nor wall-time-matched.

The frozen numerical parity gate fails:

| Seed | Learned state residual | Learned logit residual | Query predictions equal | Erase-free state residual | Erase-free logit residual |
|---:|---:|---:|:---:|---:|---:|
| 2309 | 9.54e-7 | **7.34e-4** | yes | 1.43e-6 | 4.27e-4 |
| 2311 | 7.15e-7 | 1.52e-4 | yes | 1.91e-6 | 8.25e-5 |
| 2333 | 6.56e-7 | **5.87e-4** | yes | 1.43e-6 | 3.52e-4 |

The registered bounds are `2e-6` for state and `5e-4` for logits. Thus all
state checks, every erase-free logit check, and prediction equality pass, but
learned-replay logits fail in seeds 2309 and 2333. This is consistent with
FP32 summation-order amplification through the retained RMS-normalized
decoder; it is not permission to waive the frozen integrity requirement.

## What was learned

R0--R2 showed that changing only the timing or strength of scalar symmetric
erase cannot repair these checkpoints. R3 supplies a constructive separation:
when each commissioned logical key receives an oracle component and that whole
component is reset, last-write-wins behavior appears immediately without an
optimizer update. The present learning problem is therefore not merely
"predict a bigger erase scalar." The trained dense superposition lacks an
isolated, autonomously selected replaceable unit.

The result does **not** show that the frozen two-token owner is the unique or
complete association, that a learned controller can create or select these
components, or that nine-times diagnostic state is a competitive model. It
also does not identify whether the remaining ordinary-overwrite error comes
from tail ownership, background injection, decoder coupling, finite address
separation, or their interaction.

## Next bounded move

Before fresh training, run a separately frozen ownership/coupling factorial
that preserves the successful erase-free component reset while varying only:

1. value-token-only versus the registered value-plus-`t+1` ownership;
2. following-marker tails versus filler tails;
3. background contribution at overwrite readout;
4. exact numerical reconstruction strategy, with thresholds frozen before
   quality metrics.

The decision criterion should distinguish absolute success on an already
saturated guard from improvement on the difficult ordinary-overwrite cohort.
This is a new prospective test, not a retrospective repair of R3. Only a
separate passing protocol may authorize fresh explicit-slot/occupancy training
with learned ownership and matched controls.

That successor is now frozen as the
[`G15B-R4 ownership/background factorial`](G15BR4_OWNERSHIP_BACKGROUND_PROTOCOL_2026-08-26.md).
It crosses value-only versus value-plus-tail ownership with query-time
background inclusion, binds R3 as an ineligible reference arm, and requires a
value-only pass before any training screen.

That successor is now complete. Only the two value-plus-tail arms pass; both
value-only arms fail, so no training is authorized. Excluding background does
not harm the tail-owned arm, which localizes the necessary content to the
learned continuation rather than shared-background support. See the
[`G15B-R4 result`](G15BR4_OWNERSHIP_BACKGROUND_RESULTS.md).

## Nonclaims

No parameter was trained or updated. Component ownership and reset timing use
commissioned task metadata. The constructed guard is mechanism evidence, not
generalization. Replayed G15B cells are retained-checkpoint evidence, not a
fresh cohort. Nothing here promotes G15C, the token-local controller, natural
text, an optimizer, Spin transport, scaling, or a model family.
