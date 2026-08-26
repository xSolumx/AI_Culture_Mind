# G15B-R4 ownership/background factorial results

**Completed:** 2026-08-26  
**Frozen protocol:**
[`G15BR4_OWNERSHIP_BACKGROUND_PROTOCOL_2026-08-26.md`](G15BR4_OWNERSHIP_BACKGROUND_PROTOCOL_2026-08-26.md)  
**Exact-SM75 artifact:**
[`artifacts/g15br4_ownership_background_sm75_2026-08-26.json`](artifacts/g15br4_ownership_background_sm75_2026-08-26.json)  
**Artifact SHA-256:**
`921d45e3c492e172fae62064120e9e051dca2965bacc44891268b135d8cef26e`

## Decision

**G15B-R4 fails its frozen training-authorization gate. Both value-plus-tail
arms pass, but neither value-only arm passes. Do not train the present
explicit-slot candidate: passing behavior remains dependent on ambiguous
value-plus-`t+1` ownership.**

The result localizes the retained checkpoint's useful association more
precisely than R3. `VT/BG+`, the sealed R3 reference, passes every prospective
R4 gate. `VT/BG-` also passes, showing that the shared background channel is
not required once the `t+1` injection is assigned to the key component. By
contrast, `V/BG+` fails ordinary overwrite and the constructed guard, and
`V/BG-` removes the tail from both the key component and the query read and
collapses all safety tasks. The necessary content is therefore in the learned
two-token write continuation, not in a generally useful shared background
read.

This is a zero-update retained-checkpoint diagnosis. It does not show that a
trainable controller can discover safe transaction boundaries, and the
passing `VT` arms were prospectively declared ineligible to authorize fresh
training.

## Execution contract

- clean start commit: `d014259a8956f429ae0af6326af1e45b9e2bf649`;
- clean status at start: `[]`;
- runtime: WSL Python 3.11.16, PyTorch `2.9.0+cu128`, CUDA 12.8;
- device: NVIDIA GeForce RTX 2070 SUPER, compute capability `(7,5)`;
- retained identity checkpoints: seeds 2309, 2311, and 2333;
- tasks: MQAR, overwrite, constructed overwrite guard, selective copy, and
  needle;
- lengths: 128, 512, and 1,024;
- support: 4,096 query decisions per seed/task/length cell;
- optimizer updates: zero;
- elapsed wall time: 5,031.34 seconds.

The artifact binds G15B and the sealed R3 result by SHA-256. R3 learned,
erase-free no-reset, and `VT/BG+` query metrics replay with maximum absolute
residual `0.0`.

## Frozen factorial adjudication

| Ownership | Query read | Arm | Frozen arm gate | Training-eligible |
|---|---|---|:---:|:---:|
| value only | components plus background | `v_lww_bgplus` | fail | yes, if passed |
| value only | components without background | `v_lww_bgminus` | fail | yes, if passed |
| value plus `t+1` | components plus background | `vt_lww_bgplus` | **pass** | no; sealed R3 reference |
| value plus `t+1` | components without background | `vt_lww_bgminus` | **pass** | no; tail-dependent |

`passed_value_arms=[]`, `passed_tail_arms=["vt_lww_bgplus",
"vt_lww_bgminus"]`, and `selected_training_law=null`. The overall
adjudication is therefore failed exactly as frozen.

## Three-seed mean accuracy

### Ordinary overwrite

| Length | Learned | `V/BG+` | `V/BG-` | `VT/BG+` | `VT/BG-` |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.767822 | 0.702962 | 0.491943 | **0.895752** | **0.899821** |
| 512 | 0.832764 | 0.764486 | 0.491781 | **0.954346** | **0.954183** |
| 1,024 | 0.828776 | 0.753418 | 0.494954 | **0.953613** | **0.954264** |

For the populated `after_same_key_overwrite` stratum:

| Length | Learned | `V/BG+` | `V/BG-` | `VT/BG+` | `VT/BG-` |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.754092 | 0.676990 | 0.492560 | **0.897321** | **0.902251** |
| 512 | 0.818452 | 0.737909 | 0.491350 | **0.954892** | **0.954706** |
| 1,024 | 0.815104 | 0.726190 | 0.494234 | **0.954985** | **0.955357** |

`VT/BG+` reproduces R3's 12.2--12.8-point aggregate and 13.6--14.3-point
post-same-key gains over learned erase. Background exclusion changes its
aggregate accuracy by only `+0.004069`, `-0.000163`, and `+0.000651` across
the three lengths. The key component plus its tail already contains the
useful decoded association.

Value-only replacement is not merely a near miss hidden by averaging. In seed
2311, `V/BG+` aggregate accuracy is `0.364746`, `0.425049`, and `0.392090`,
versus learned `0.729980`, `0.797607`, and `0.785889`. Seeds 2309 and 2333 are
stronger but do not satisfy the full frozen per-seed conjunction. The
checkpoint family has no multi-seed robust value-local replacement law.

The ordinary generator again contains no unrelated-overwrite-only queries, so
no ordinary-cohort claim is made for that stratum.

### Constructed all-strata guard

| Length | Learned | `V/BG+` | `V/BG-` | `VT/BG+` | `VT/BG-` |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.999756 | 0.938965 | 0.624919 | **1.000000** | **1.000000** |
| 512 | 0.999919 | 0.939697 | 0.633138 | **1.000000** | **0.999919** |
| 1,024 | 0.999674 | 0.941813 | 0.630859 | **1.000000** | **1.000000** |

Every length contains 3,072 queries before any overwrite, 4,608 after only an
unrelated overwrite, and 4,608 after a same-key overwrite. `VT/BG+` is exactly
1.0 in every stratum. `VT/BG-` differs only once: its L512 post-same-key
accuracy is `0.999783`; it still passes every ceiling-aware guard gate.

`V/BG+` fails the registered absolute `0.995` aggregate and post-same-key
floors. Its post-same-key values are `0.838759`, `0.839627`, and `0.844835`.
This closes R3's saturated-baseline ambiguity prospectively: the value-only
law fails an absolute guard criterion, independent of an impossible
improvement over an already perfect learned reference.

### Safety tasks

`V/BG+` and `VT/BG+` are identical on the no-repeat safety tasks and satisfy
their frozen gates: MQAR is `0.980306`--`0.983643`, selective copy is
`0.960775`--`0.965658`, and needle is exactly `1.0`. `VT/BG-` remains within
the same gates. `V/BG-` falls to roughly `0.56` MQAR, `0.52` selective copy,
and `0.64` needle and therefore fails all nine safety cells. This is expected
under the now-observed ownership law: with value-only ownership, the useful
`t+1` continuation remains in background, and `BG-` removes it at the query.

## Integrity result

All frozen integrity gates pass:

- exact R3 reference replay residual: `0.0`;
- ordinary model-forward logit residual: `0.0` in every seed;
- no-reset query predictions: equal for both ownership rules;
- LWW/no-reset equality on all no-overwrite tasks and pre-overwrite prefixes:
  true for every arm;
- preserved non-erase controls: bitwise equal;
- finite logits: true;
- FP64 monolithic/component/recurrent/parallel maximum residual:
  `4.44e-15`, below the frozen `1e-10` bound.

FP32 decomposition state residuals are at most `1.91e-6`, and the largest
recorded background-read identity residual is `3.68e-6`; all replay
predictions are identical. R4's FP64 algebraic proof is separately
preregistered and does not waive or relabel R3's failed FP32 logit threshold.

Across seeds, 413 final-token writes are recorded without invented tails.
In-range `t+1` owners include 220,054 filler tokens, 13,468 following write
markers, and 1,073 item markers. Thus the required continuation is not
semantically equivalent to "own the following filler." It sometimes claims
the beginning of a subsequent event, which is precisely why `VT` is not a
clean training law.

The eight-key decomposition remains nine times the base recurrent state:
2,304 versus 256 scalars per sequence. The two-key guard uses 512 versus 256.
This diagnostic is not state-, parameter-, compute-, or wall-time-matched.

## What was learned

R0--R2 rejected scalar symmetric erase. R3 showed that resetting a complete
oracle component can restore last-write-wins behavior. R4 now shows what that
component contains in these checkpoints: a learned, seed-sensitive two-token
write transaction. The value-token injection alone is not the replaceable
association, and query-time background is not intrinsically necessary once
the continuation is attached to the key.

The present learning problem is therefore **transaction formation and causal
ownership**, before optimizer choice. A useful new model must learn an
explicit, locally decidable begin/accumulate/commit law that prevents one
event's continuation from ambiguously consuming the next event's marker. It
must also learn its addresses and occupancy rather than receive oracle logical
keys. Changing AdamW, learning-rate decay, Spin transport, or the tokenizer
does not repair this missing state law in the retained checkpoints.

## Next bounded move

Do not launch the explicit-slot training screen registered after R3. First
freeze a tail-localization diagnostic that asks whether the required `t+1`
term can be replaced by a causal pending-write/commit variable derived only
from the preceding value event, without assigning the current token's own
injection to the previous key. The decisive control must preserve `VT`
performance while preventing marker ownership overlap. If no such locally
decidable transaction law survives fresh seeds, stop this commissioned
checkpoint-repair branch and move to a separately preregistered GDN2/KDA-like
model with explicit transaction and occupancy state trained from scratch.

## Nonclaims

No parameter was trained or updated. Component ownership uses commissioned
task metadata. Replayed schedules are not fresh generalization. Passing `VT`
does not establish autonomous memory learning, generic association,
ordinary-text recall, longer-context scaling, efficiency, optimizer or
tokenizer superiority, Spin benefit, G15C, or model-family promotion. Failing
value-only post-hoc replacement does not falsify explicit-slot, GDN2, KDA, or
dual-address memories trained with a different state law.
