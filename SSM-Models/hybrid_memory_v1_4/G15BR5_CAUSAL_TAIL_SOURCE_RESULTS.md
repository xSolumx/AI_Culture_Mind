# G15B-R5 causal tail-source results

**Frozen protocol:**
[`G15BR5_CAUSAL_TAIL_SOURCE_PROTOCOL_2026-08-26.md`](G15BR5_CAUSAL_TAIL_SOURCE_PROTOCOL_2026-08-26.md)

**Exact artifact:**
[`artifacts/g15br5_causal_tail_source_sm75_2026-08-26.json`](artifacts/g15br5_causal_tail_source_sm75_2026-08-26.json)

**SHA-256:**
`ba627fe34e8dd29458fc1321b52c98242838c3b56e2abdc7e44c749f50aaa313`

**Execution commit:** `e039e499b44b8e9bbb1108eb456c051a4702ba4e`
**Status:** frozen formal fail; performance-positive history attribution; no
training authorization pending a prospective numerical ratification

## Bottom line

R5 found the first clean causal source for the association that R4 had left
ambiguous:

> the strict three-token convolution history at `t+1`, containing the completed
> `[write/select marker, key, value]` transaction, is sufficient for the
> replaceable association when residual background is excluded at queries.

`h_lww_bgminus` passed all **132/132** frozen performance, guard, safety,
per-seed, and bias-separation checks. Current-only and bias-only arms failed.
The history result is conditional on the ordinary full-token transition, which
remains current-token dependent; it is not a wholly history-only update.

The frozen overall adjudication nevertheless failed. Two numerical-integrity
rules were stricter than the algebra they were checking:

1. every discrete R4 replay metric was exact, but the decomposed no-reset
   `bits_per_query` differed by at most `1.3974e-7` from R4 against a frozen
   `1e-12` bound;
2. FP32 component summation reached `2.3842e-6` state residual and `3.5763e-6`
   read-relation residual against a frozen `2e-6` absolute bound.

Learned logits were bit-identical, every no-reset reference prediction was
identical, every no-overwrite/prefix prediction contract passed, source
assignment was exact, and the independent FP64 algebraic maximum was
`3.9968e-15`. These facts strongly identify FP32 non-associativity rather than
a semantic recurrence mismatch. The formal R5 result is not relabeled after
seeing it: its artifact remains a fail and authorizes no training. A separate,
prospectively frozen R5-S numerical ratification is the next gate.

## Exact execution

The quality run used:

- Python 3.11.16;
- PyTorch 2.9.0+cu128;
- NVIDIA GeForce RTX 2070 SUPER;
- exact compute capability `(7,5)`;
- clean Git status at start;
- seeds 2309, 2311, and 2333;
- 4,096 query decisions per task/length/seed cell;
- lengths 128, 512, and 1,024;
- zero optimizer updates;
- `4,611.91` seconds wall time.

The artifact binds the original G15B parent at
`f74d860e30ab40ec747521dfcecd74aac2bb75151206c25b7104d334727429eb`
and R4 at
`921d45e3c492e172fae62064120e9e051dca2965bacc44891268b135d8cef26e`.
All batch fingerprints matched R4.

## Ordinary overwrite result

Three-seed mean query accuracy:

| Length | Learned | H no-reset BG- | H LWW BG+ | **H LWW BG-** | C LWW BG- | B LWW BG- |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.767822 | 0.385173 | 0.864258 | **0.942383** | 0.483317 | 0.493652 |
| 512 | 0.832764 | 0.402425 | 0.943441 | **0.945557** | 0.495443 | 0.492269 |
| 1,024 | 0.828776 | 0.389323 | 0.942708 | **0.946615** | 0.496338 | 0.495768 |

Three-seed mean accuracy after a same-key overwrite:

| Length | Learned | H no-reset BG- | H LWW BG+ | **H LWW BG-** | C LWW BG- | B LWW BG- |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.754092 | 0.306734 | 0.861328 | **0.943545** | 0.484840 | 0.494234 |
| 512 | 0.818452 | 0.324684 | 0.942429 | **0.945406** | 0.495257 | 0.491908 |
| 1,024 | 0.815104 | 0.310919 | 0.942522 | **0.947824** | 0.496559 | 0.496187 |

The background-free history arm beat learned ordinary accuracy by
`+0.1746`, `+0.1128`, and `+0.1178` in the three-seed means. It beat its
matched no-reset control by more than `+0.54` at every length. It also passed
the frozen within-seed improvements over learned, no-reset, and bias at all
three lengths. The narrowest per-seed bias margin was seed 2309, where the
aggregate margins were `+0.0295`, `+0.0264`, and `+0.0288`; all exceed the
frozen `+0.02` per-seed gate.

`h_lww_bgplus` passed 125/132 checks but failed seven. It missed the L128 mean
improvement over learned and failed the seed-2309 bias-separation checks at all
lengths. Residual background is therefore not merely unnecessary: for this
checkpoint it can hide the clean history attribution.

## Constructed guard and safety

`h_lww_bgminus` achieved exactly `1.0` mean aggregate and post-same-key
accuracy on the constructed overwrite guard at every length. Learned was
already between `0.999674` and `0.999919`; the H arm preserved that ceiling
while the matched no-reset control remained near `0.80` aggregate and `0.47` to
`0.49` post-same-key.

| Length | H BG- MQAR | Learned MQAR | H BG- selective | Learned selective | H BG- needle |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.977865 | 0.972412 | 0.955566 | 0.971924 | 1.000000 |
| 512 | 0.979085 | 0.972900 | 0.957764 | 0.976969 | 1.000000 |
| 1,024 | 0.982178 | 0.971842 | 0.959961 | 0.975179 | 1.000000 |

All frozen MQAR, selective-copy, needle, before-overwrite, and guard-stratum
safety checks passed for `h_lww_bgminus`.

## Why history is the identified source

The source statistics agree with the intervention result. Across filler tails,
mean history-source injection norms were `0.854`, `1.930`, and `1.268` for the
three checkpoints, close to the corresponding full-injection norms `0.839`,
`1.931`, and `1.320`. Bias-only norms were only `0.033`, `0.028`, and `0.028`;
current-only norms were `0.037`, `0.035`, and `0.077`.

Current-only and bias-only background-free arms stayed near chance on ordinary
overwrite and around `0.63` on the constructed guard. Their BG+ variants also
failed dozens of gates. The useful tail association therefore does not come
from convolution bias or the new token by itself. It comes from the completed
transaction in the strict causal history.

This is a retained-checkpoint sufficiency result. The history address is still
assigned to oracle logical-key components, and the checkpoint never learned an
autonomous pending-write state.

## Formal integrity failure

The frozen artifact reports:

| Contract | Observed maximum | Frozen bound/status |
|---|---:|---:|
| Learned-logit replay | 0 | exact pass |
| R4 learned accuracy/episode/BPQ replay | 0 | exact pass |
| R4 no-reset accuracy/episode replay | 0 | exact pass |
| R4 no-reset BPQ replay | `1.397359e-7` | fail vs `1e-12` |
| FP32 preactivation reconstruction | `1.907349e-6` | pass vs `2e-6` |
| FP32 injection sum | `5.960464e-8` | pass |
| FP32 source assignment | 0 | exact pass |
| FP32 no-reset state sum | `2.384186e-6` | fail vs `2e-6` |
| FP32 background read relation | `3.576279e-6` | fail vs `2e-6` |
| Independent FP64 algebra | `3.996803e-15` | pass vs `1e-10` |

The worst BPQ mismatch was R4 `erase_free_no_reset_bgplus` at seed 2309,
overwrite L1024: `4.1686114794` versus `4.1686116192`, a relative difference of
`3.35e-8`. The learned path replayed exactly. All recorded no-reset query
predictions were identical to the direct monolithic reference. All source
assignments, shared transitions, causal locality witnesses, chunk/impulse
checks, finite-logit checks, and LWW prefix invariants passed.

The `2e-6` FP32 bound had been calibrated on a 16-decision smoke before the
quality run. The complete quality grid exposed longer reduction paths and
larger state magnitudes. Changing that bound after seeing the result would be a
post-hoc promotion, so R5 remains formally failed.

## Decision and next move

The frozen artifact's exact decision is:

> `stop retained-checkpoint tail repair; no bias-separated background-free history source passes`

That string follows from the overall pass list being emptied by replay/runtime
integrity. It must not be quoted as if `h_lww_bgminus` failed its performance or
bias gates; it did not.

The next move is **R5-S**, a separate prospective numerical ratification:

1. bind this artifact and leave its formal result unchanged;
2. use fresh evaluation batches and scale-aware IEEE-FP32 error bounds fixed
   before execution;
3. compare direct monolithic, decomposed FP32, and independently recomputed
   FP64 no-reset states, reads, logits, predictions, and NLL;
4. require exact discrete replay and source assignments;
5. authorize a fresh pending-write/commit training protocol only if numerical
   stability passes without changing any R5 performance metric.

If R5-S fails, stop the retained-checkpoint route. If it passes, the next model
should learn an explicit causal transaction accumulator and protected
background-free transaction read on fresh seeds, with parameter/state/compute-
matched controls. That would still be a training screen, not model promotion.

## Nonclaims

R5 does not establish autonomous transaction learning, generic association,
natural-text recall, longer-context scaling beyond 1,024, efficiency,
optimizer/tokenizer superiority, Spin benefit, G15C, or model-family
promotion. It uses expanded oracle logical components and is not parameter-,
state-, compute-, or wall-time matched. The signed source residual remains an
attribution tool, not a proposed bounded learned write law.
