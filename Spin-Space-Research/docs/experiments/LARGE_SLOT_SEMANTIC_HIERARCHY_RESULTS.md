# Large-slot semantic hierarchy and fused gathered memory results

- **Programme:** Triality memory and Intertwiner SchurScans
- **Date:** 2026-08-10
- **Quality protocol:**
  [LARGE_SLOT_SEMANTIC_HIERARCHY_PREREGISTRATION.md](LARGE_SLOT_SEMANTIC_HIERARCHY_PREREGISTRATION.md)
- **Eager systems protocol:**
  [GATHERED_BLOCK_MEMORY_BENCHMARK_PREREGISTRATION.md](GATHERED_BLOCK_MEMORY_BENCHMARK_PREREGISTRATION.md)
- **Fused systems protocol:**
  [FUSED_GATHERED_BLOCK_MEMORY_PREREGISTRATION.md](FUSED_GATHERED_BLOCK_MEMORY_PREREGISTRATION.md)
- **Verdict:** all three frozen decisions pass; the strongest result is a shared
  coarse-to-fine router feeding an actual fused sparse-state recurrence, not a
  new memory law or extra Spin(8) storage capacity

## Executive result

The programme now has a complete local evidence chain:

1. supplied Spin(8) inverse-frame transport canonicalizes three views;
2. one shared 576-parameter router completes label/view combinations omitted
   from each independent router's training set;
3. coarse-to-fine routing improves long-stream retrieval over a same-router
   dense soft route in a 64-slot overlapping-semantic world;
4. physically gathering the selected `8 x 8` state block preserves the eager
   recurrence within float32 tolerance;
5. fusing routing, gathered write, and gathered read into one Triton kernel
   removes the eager launch bottleneck and beats both eager controls.

This is the first campaign here that joins learned large-slot routing, recurrent
quality, physical sparse state access, and measured CUDA latency. It does not
show that direct or delta updates have a triality-specific capacity advantage.

## Frozen decisions

| Decision | Required | Observed | Verdict |
|---|---:|---:|---:|
| quality implementation gates | every seed | 10/10 | pass |
| shared-router completion | at least 8/10 | 10/10 | pass |
| hierarchy improves direct and delta | at least 8/10 | 10/10 | pass |
| eager gather beats masked-full, principal cells | direct and delta | 4/4 | pass |
| fused gather beats eager dense and gather, principal cells | direct and delta | 4/4 | pass |

The development seed `103` was excluded. The quality decision used untouched
seeds `30`--`39`; the eager and fused systems decisions each used three
independent measurement processes.

## Learned semantic routing

The world contains 64 correlated fine keys arranged in eight semantic blocks,
three transported views, and four Spin(8) action words. Action word `3` is held
out. Label `k` is omitted from training view `k mod 3`, so an independent router
has no positive example for roughly one third of its label/view combinations.
The shared router sees the same examples but pools them across views after the
same supplied inverse-frame canonicalization.

Across frozen seeds:

- shared held-out hard-route accuracy: mean `0.909180`, range
  `0.883789`--`0.926758`;
- independent held-out hard-route accuracy: mean `0.000000`;
- independent observed accuracy: mean `0.943945`;
- shared router parameters: `576`; independent control: `1728`;
- maximum canonicalization error: `2.33e-15`;
- maximum retained-parameter replay difference: `2.22e-16`.

The zero independent held-out result is a direct consequence of the frozen
missing-label design and therefore should not be read as a generic indictment
of independent models. What the result establishes is narrower and useful:
weight sharing completes missing label/view cells when a valid common frame is
supplied, even against a three-times-larger independent control.

Mean hard-route accuracies reveal the noise frontier:

| Alias radius | Shared observed | Shared held out | Independent observed | Independent held out |
|---:|---:|---:|---:|---:|
| `0.20` | `0.961914` | `0.960742` | `0.981250` | `0.000000` |
| `0.40` | `0.854297` | `0.857617` | `0.906641` | `0.000000` |
| `0.60` | `0.648047` | `0.648047` | `0.735352` | `0.000000` |

## Long-stream memory quality

The frozen hierarchy decision averages the shared router's length-2048,
radius-`0.20` and radius-`0.40` improvements. `block_top1` beats `dense_soft`
on every seed for both update laws:

- direct mean improvement `0.096836`, range
  `0.092296`--`0.101359`;
- delta mean improvement `0.045765`, range
  `0.042526`--`0.048516`.

The full length-2048 held-out-view frontier is:

| Radius | Direct dense | Direct block | Direct hard | Delta dense | Delta block | Delta hard |
|---:|---:|---:|---:|---:|---:|---:|
| `0.20` | `0.215983` | `0.307797` | `0.913010` | `0.239257` | `0.279877` | `0.913010` |
| `0.40` | `0.197924` | `0.299735` | `0.662089` | `0.204261` | `0.254920` | `0.662089` |
| `0.60` | `0.173041` | `0.271724` | `0.342463` | `0.159791` | `0.205792` | `0.342463` |

These are mean query cosines. The complete 983,040-query frozen quality cohort
also retains observed-view results, lengths `256` and `1024`, minimum cosines,
relative squared errors, and every learned parameter.

Hard one-hot routing makes direct and normalized delta memory mathematically
the same addressed overwrite. The observed maximum state and prediction
difference was `1.11e-16`. Soft-route differences therefore concern interference
geometry; they do not establish a superior update law.

## Eager gathered-state benchmark

The first CUDA protocol separates logical sparsity from physical execution.
`block_masked_full` constructs a full-slot route; `block_gathered` mutates and
reads only the selected `8 x 8` block. Correctness passes with maximum state
error `1.19e-7` and prediction error `1.19e-7`.

At batch 16:

| Slots | Law | Masked full ms | Gathered ms | Speedup | Masked bytes | Gathered bytes |
|---:|---|---:|---:|---:|---:|---:|
| 1,024 | direct | `0.999323` | `0.798756` | `1.251x` | 657,408 | 18,944 |
| 1,024 | delta | `1.307481` | `1.038122` | `1.259x` | 723,968 | 15,872 |
| 4,096 | direct | `0.974091` | `0.777443` | `1.253x` | 2,623,488 | 34,304 |
| 4,096 | delta | `1.233692` | `1.005326` | `1.227x` | 2,886,656 | 34,304 |

The frozen masked-full comparison passes, but the eager gather is only
`0.481x`--`0.645x` as fast as eager dense over the complete 12-cell grid. The
selected-state arithmetic was real; the many small PyTorch launches dominated
latency. This negative boundary was retained and motivated a separately frozen
fusion protocol.

## Fused gathered-state benchmark

A single `triton-windows==3.7.1.post27` kernel now performs coarse routing,
eight-way fine routing, the selected direct or delta update, and selected read.
Across three new processes, maximum eager/fused state error is `1.79e-7` and
maximum prediction error `1.79e-7`.

Principal batch-16 results:

| Slots | Law | Fused ms | vs eager dense | vs eager gather | Incremental bytes |
|---:|---|---:|---:|---:|---:|
| 1,024 | direct | `0.051315` | `7.747x` | `15.726x` | 512 |
| 1,024 | delta | `0.051067` | `12.665x` | `20.946x` | 512 |
| 4,096 | direct | `0.053683` | `7.694x` | `15.618x` | 512 |
| 4,096 | delta | `0.052836` | `12.696x` | `20.721x` | 512 |

The full grid is consistent rather than resting on the principal cells:

- direct versus eager dense: `7.602x`--`8.118x`;
- direct versus eager gather: `15.537x`--`16.361x`;
- delta versus eager dense: `12.209x`--`13.000x`;
- delta versus eager gather: `20.035x`--`21.399x`;
- fused latency: approximately `49.9`--`53.7` microseconds.

### Timing-order correction before publication

A final code audit found that the first implementation rotated variant order
once per cell rather than across timing blocks as required by the eager frozen
protocol. The scheduler was corrected so every block cyclically changes the
six-variant order while each variant retains its own recurrent state. All three
eager processes, all three fused processes, both development timings, and both
aggregates were regenerated. Every reported number above comes only from the
corrected artifacts, which retain `timing_block_orders` for audit. The frozen
grid, warmup, call counts, correctness gates, and decisions were unchanged.

The allocation numbers are PyTorch allocator deltas, not total working-set
memory or hardware traffic. They nevertheless verify that the fused call does
not materialize a slot-scaled intermediate in the tested path.

## Recurrent strengthening check

After the frozen fused decision, a separate diagnostic repeated the identical
input 257 times at batch 16 and 64, 1,024, and 4,096 logical slots. All six
law/slot cells passed `1e-5`:

- worst state error: `3.16e-6`;
- worst final prediction error: `4.47e-8`;
- all outputs finite.

This is evidence against immediate recurrent numerical drift. It is explicitly
not a preregistered decision gate, does not vary the input sequence, and does
not test gradients.

## Relation to Native Sparse Attention

Native Sparse Attention separates coarse compressed selection, selected
fine-grained access, and a local window. The transferable lesson for this
programme is architectural: do not make a single dense soft key carry semantic
identity, overwrite discrimination, and retrieval. The present experiment
implements the coarse-to-fine selected-memory portion with a recurrent
SchurScan-compatible state update. It does not reproduce NSA's learned token
compression, local window, end-to-end language-model training, or production
kernels.

The remaining high-value model experiment is therefore a matched three-branch
system: local recent window, fused selected block memory, and a compressed
global summary, with a learned gate and identical training/compute budgets.
Spin(9) should enter only as a separately falsifiable binding or routing prior;
larger Clifford structure alone is not evidence of more usable memory.

## Claim boundary

Established on the frozen synthetic world and named GPU:

- shared supplied-frame routing completes missing view cells;
- learned block selection reduces soft-address interference;
- direct and delta memories coincide under the same hard address;
- physical gathered-state inference can be correct and fast when fused.

Not established:

- learned Spin(8) action or frame discovery;
- triality-specific storage capacity;
- a direct-over-delta or delta-over-direct theorem for soft routes;
- Spin(9) superiority;
- training-kernel, language-model, cross-GPU, or production-system superiority.

## Reproduction and artifacts

Primary records:

- `artifacts/large_slot_semantic_hierarchy_seeds30_39.json`;
- `artifacts/gathered_block_memory_cuda_aggregate_20260810.json`;
- `artifacts/fused_gathered_block_memory_cuda_aggregate_20260810.json`;
- `artifacts/fused_gathered_block_memory_recurrent_diagnostic_20260810.json`.

The aggregates hash all contributing process/seed artifacts. Exact commands
are listed in [REPRODUCIBILITY.md](../REPRODUCIBILITY.md), and every retained
artifact is covered by `ARTIFACTS.sha256`.
