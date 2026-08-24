# Hybrid Memory SSM v1.4

**Status:** v1.4.4 tied-address successor passed prospectively frozen G4f with
internal commissioning labels. The external causal reverse-binding route
passed the G8 continuation but failed the fresh-from-scratch G9 gate: one of
three fresh seeds fell below 90% at both L96 and L512. Diagnostics localize
that failure to learned global filler decay, not a disconnected learning
signal. A retention-only v1.4.5 candidate repaired the exposed weak seed, but
that is development evidence pending fresh validation. The frozen G4a
selected-memory result remains negative; none of these results is a label-free,
natural-language, or speed promotion.

This track combines five explicit mixer kinds in one causal language-model
shell:

- bounded sliding-window RoPE attention with a complete streaming KV cache;
- content-addressed Gated DeltaNet fast-weight memory;
- the maintained repository DeltaProduct reference;
- hierarchical selected-block affine memory;
- bounded rung-routed Spin(8) memory.

Layer schedules are explicit `layer_plan` tuples. Full-sequence, arbitrary
chunk, and token-step execution carry complete recurrent, convolution, and
attention state. `state_byte_report` reports actual and configured-capacity
bytes rather than an advertised partial cache.

## The important negative result

The original preregistered G3 claim was false. Always-live reads do not make a
non-final read affect a final-only loss in a one-block model. Coarse block
selection was also a hard `argmax`, so retrieval loss could not train its
logits.

The failure is retained as evidence, not reworded as a success. The successor
mechanism adds explicit dense `soft` and `straight_through` route modes while
keeping sparse `physical_gather` hard-only. Deterministic float64 replay now
separates three facts:

1. hard one-block routing has disconnected coarse logits and zero non-final
   read credit;
2. straight-through routing connects coarse logits but cannot change the
   one-block temporal dependency, so non-final read credit remains zero;
3. a `selected_block -> attention` schedule with straight-through training
   has a real non-final coarse and fine read path to final-only loss.

See [`RESULTS.md`](RESULTS.md),
[`PREREGISTRATION.md`](PREREGISTRATION.md), and the checked artifact
[`artifacts/temporal_observability_2026-08-24.json`](artifacts/temporal_observability_2026-08-24.json).

The subsequent three-seed, 600-update G4a MQAR campaign also failed. The
label-free candidate reached only 1.563%--3.125% L512 accuracy per seed; the
explicitly supervised-routing ablation reached 1.172%--2.344%. The matched
DeltaProduct shell also remained near chance, so this falsifies the frozen
capability claim without isolating one memory architecture as the cause. See
[`RESULTS.md`](RESULTS.md) for the complete adjudication.

## v1.4.1 learning pivot

The deeper G4a audit found that selected memory stored values but no key
signatures. Its 16 static addresses per head had a last-write-wins oracle
ceiling below the frozen 90% gate on 16-pair MQAR. G4a also supplied only
19,200 scored query labels; its large token count was dominated by filler.

The v1.4.1 default is now `gated_delta -> attention`. The Gated Delta state
stores key-to-value bindings in a matrix, with learned write strength and
retention. Exact recurrent and parallel affine scans, arbitrary-chunk replay,
masked-state semantics, gradient paths, and cache bytes are tested.

The development curriculum reached 94.34% at length 96 and 92.68% over a
larger unseen 2,048-query length-512 cohort. A fresh length-512 continuation
raised that cohort to 94.19%. Association and write events were explicitly
supervised from synthetic task metadata, so these results prove commissioned
rule learning, not label-free discovery. See
[`LEARNABILITY_DIAGNOSIS.md`](LEARNABILITY_DIAGNOSIS.md) and the prospectively
frozen [`G4B_PREREGISTRATION.md`](G4B_PREREGISTRATION.md).

G4b subsequently failed its all-seed gate: seeds 1423 and 1427 reached 96.14%
and 94.14% at length 512, while seed 1429 reached 84.81%. The mean was 91.70%,
but an average cannot hide the weak seed. The uniform target-length
consolidation test is frozen in
[`G4C_PREREGISTRATION.md`](G4C_PREREGISTRATION.md); G4b remains failed whatever
G4c reports.

G4c also failed: identical consolidation moved weak seed 1429 only to 87.60%
at L512. Cross-seed diagnostics then showed perfect address top-1 accuracy but
seed-dependent memory interference. v1.4.2 doubles key dimension, normalizes
value injection, and directly supervises the post-memory readout. It reached
97.17% and 92.68% at L512 on already exposed development seeds 1401 and 1429.
Fresh seeds 1451/1453/1459 and the all-seed 90% gate are frozen in
[`G4D_PREREGISTRATION.md`](G4D_PREREGISTRATION.md).

G4d failed because fresh seed 1459 reached 86.33% at L512. v1.4.3 removed the
random value/readout basin with identity value and output projections, an
identity-centered output gate, and a 0.5 initial memory residual. G4e still
failed: seed 1481 reached only 78.22% at L512 while the other two seeds reached
99.32% and 97.17%. Its query/key projections had fallen into different address
frames: per-head address accuracy was 97.7-99.8% and mean margin 0.22, versus
100% and about 0.42 in the strong seeds.

v1.4.4 removes that independent-frame lottery by using one tied, orthogonally
initialized query/key projection. On the already exposed failed seed 1481 it
reached 97.998% at L512 in development. The prospectively frozen G4f seeds
1511/1523/1531 then all passed the 90% gate at both lengths. Mean exact query
accuracy was 98.95% at L96 and 98.67% at L512; the minima were 98.72% and
98.10%. See [`G4F_PREREGISTRATION.md`](G4F_PREREGISTRATION.md) and the retained
validation artifact. This is validated synthetic associative learning, but it
remains label-supervised with 774,400 useful query labels per seed.

## Implemented surfaces

- [`tasks.py`](tasks.py): causal MQAR, overwrite retrieval, exact-distance
  needle retrieval, and selective copy. Scored answers are absent from query
  input positions.
- [`selected_block.py`](selected_block.py): bounded affine memory with
  independent write/erase/read routes; sparse hard gather/scatter; dense hard,
  soft, and straight-through semantic modes; recurrent/parallel oracles.
- [`gated_delta.py`](gated_delta.py): content-addressed matrix state with exact
  recurrent/parallel semantics, learned write and retention controls, and
  complete streaming state.
- [`attention.py`](attention.py): bounded causal local attention and cache.
- [`structured_tier.py`](structured_tier.py) and
  [`structured_memory.py`](structured_memory.py): data-routed subgroup ladder
  and a recurrent Spin(8) state that actually changes the model output.
- [`model.py`](model.py): five-kind hybrid shell, local convolution caches,
  checkpointing, diagnostics, and exact cache-byte accounting.
- [`fla_adapter.py`](fla_adapter.py): fail-closed semantic and optional official
  FLA DeltaRule operator adapters.
- [`baselines.py`](baselines.py): precise implementation registry and claim
  boundaries. Static ProductKey memory is explicitly non-episodic; FLA
  adapters are operators, not complete language models; FlashRT fails closed
  on SM75; actual Transformers Mamba-2 and OLMo Hybrid implementations have
  separately labeled unfused rows.
- [`audits.py`](audits.py): precision horizon, complete chunk replay, cache
  drift, structured rung/gauge, and temporal observability audits.
- [`experiments.py`](experiments.py): fresh deterministic episodes, paired
  cohorts, pre/post-training evaluation, parameter gate, explicit route
  training mode, source hashes, dirty/untracked Git status, state-size sweeps,
  and atomic JSON-compatible schemas.
- [`retrieval_screen.py`](retrieval_screen.py): exact non-evidentiary smoke and
  frozen G4a matched MQAR campaign builders plus gate adjudication.
- [`long_context_screen.py`](long_context_screen.py): non-training streaming
  screen at configurable lengths.
- [`temporal_observability_screen.py`](temporal_observability_screen.py):
  deterministic replay of the failed original G3 and accepted successor
  topology.
- [`precision_screen.py`](precision_screen.py): direct-float64 fp16/fp32 error
  curves through length 65,536 for generic affine, selected-block, and
  maintained DeltaProduct transitions.
- [`learnability_screen.py`](learnability_screen.py),
  [`long_context_continuation.py`](long_context_continuation.py), and
  [`validation_screen.py`](validation_screen.py): fixed-batch falsifier and the
  original useful-label-accounted curriculum.
- [`successor_screen.py`](successor_screen.py),
  [`identity_validation.py`](identity_validation.py), and
  [`tied_validation.py`](tied_validation.py): interference, identity-path, and
  tied-address successor experiments with retained checkpoints and
  prospectively frozen fresh-seed gates.
- [`upstream_probe.py`](upstream_probe.py): actual FLA, Transformers, and
  pretrained state-spaces Mamba-2 probes with explicit environment boundaries.
- [`upstream_learning_comparison.py`](upstream_learning_comparison.py): paired
  learning screens using the actual Transformers Mamba-2 and OLMo Hybrid
  implementations, with external-label and single-seed boundaries explicit.
- [`reverse_binding_validation.py`](reverse_binding_validation.py),
  [`competence_validation.py`](competence_validation.py), and
  [`distance_consolidation.py`](distance_consolidation.py): fresh-seed external
  learning, competence-pacing falsification, and retained-optimizer L512
  consolidation.
- [`combined_validation.py`](combined_validation.py) and
  [`optimization_diagnostic.py`](optimization_diagnostic.py): the frozen G9
  fresh combined schedule and causal/representation diagnostics for its weak
  seed.
- [`retention_successor_screen.py`](retention_successor_screen.py): the matched
  exposed-seed retention-only causal intervention for the v1.4.5 candidate.

## Baseline boundary

The normal matched quality control is a `HybridMemoryLM` with the same shell
and a `delta_product` layer plan. A static ProductKey table is not a fair
episodic MQAR control. The FLA DeltaRule registry entries are operator-level
semantic/systems controls, not silently promoted language models. Official
Mamba-2 is an optional separate complete-model comparison and fails closed
when unavailable.

The exact pretrained `state-spaces/mamba2-130m` weights are pinned at revision
`3a5aea0c25d0fb43cc360e2c2aac82c26e3eed49` in the external E: cache and were
loaded through official `mamba_ssm` in WSL. They are not committed to Git.

G5 also trained small actual Transformers Mamba-2 and OLMo Hybrid models on the
same synthetic episode stream as v1.4.4. Its proposed next-token write target
was irreducible noise because synthetic values are random and unseen. With the
remaining retrieval signal, OLMo learned L96 (97.52%) but transferred poorly
to L512 (25.44%); Mamba-2 reached 17.97%/14.16%, and uncommissioned v1.4.4
5.69%/4.44%. G5b replaced the noise with causal reverse-binding reconstruction
at the observed value event. All three models then learned L96; at L512,
v1.4.4 reached 96.00%, Mamba-2 77.98%, and OLMo Hybrid 58.11%. This is one
paired seed, not an upstream superiority claim. A three-fresh-seed v1.4.4 gate
was frozen in [`G6_PREREGISTRATION.md`](G6_PREREGISTRATION.md).

G6 failed because seed 1643 reached only 89.21% L96 and 83.40% L512. Its
reverse-binding auxiliary was learned, but the fixed curriculum advanced while
retrieval accuracy was still 12.5%, 17.2%, and 53.9% at the first three phase
boundaries. G7 freezes a competence-paced curriculum: two consecutive fresh
probes must clear 90% before a phase advances, subject to fixed caps. See
[`G7_PREREGISTRATION.md`](G7_PREREGISTRATION.md).

G7 failed too. Every seed exceeded 94% at L96, but L512 mean/minimum were only
89.40%/85.89%; two seeds also missed early-phase competence caps before later
mastering harder phases. Phase mastery is not monotonic, and L96 mastery is not
retention evidence. G8 froze one uniform 600-update L512 consolidation phase
for every G7 checkpoint, with optimizer state restored and no seed-specific
tuning. It passed: L96 mean/minimum were 95.63%/93.69%, and L512 mean/minimum
were 94.61%/92.29%. See
[`G8_PREREGISTRATION.md`](G8_PREREGISTRATION.md).

G9 froze the complete external schedule from random initialization on three
unseen seeds. It failed: seeds 1721 and 1733 reached 98.54% and 95.36% at L512,
but seed 1723 reached only 87.89% (and 88.27% at L96). The weak checkpoint had
one head writing at 99.76% strength on filler while retaining only 90.04% per
filler token. Even 416 such transitions erase essentially the entire old-state
contribution. Retrieval gradients remained nonzero and removing the Gated
Delta layer collapsed accuracy, so the present problem is unstable learned
retention, not absent credit or an unused memory layer. See
[`G9_PREREGISTRATION.md`](G9_PREREGISTRATION.md) and the retained G9 artifacts.

The v1.4.5 development candidate changes only minimum/initial retention from
0.90/0.995 to 0.999/0.9995. On the exact seed-1723 replay it raised L96 from
88.27% to 98.06% and L512 from 87.89% to 96.29%. In a separate diagnostic
cohort it reached 98.24%; ablating Gated Delta reduced accuracy to 0%, while
ablating attention left 98.24%. Active-head filler retention was 0.99973.
Because seed 1723 was already exposed, this is causal development evidence,
not validation.

G10 prospectively freezes the same retention-safe candidate and complete G9
schedule on unseen seeds 1753/1759/1777. Every seed must reach at least 90% at
both lengths. See [`G10_PREREGISTRATION.md`](G10_PREREGISTRATION.md).

## Validation

From `SSM-Models`:

```powershell
python -m pytest hybrid_memory_v1_4/tests -q
python -m hybrid_memory_v1_4.temporal_observability_screen --output hybrid_memory_v1_4/artifacts/temporal_observability_2026-08-24.json
python -m hybrid_memory_v1_4.precision_screen --output hybrid_memory_v1_4/artifacts/precision_horizon_65536_cpu_2026-08-24.json
python -m hybrid_memory_v1_4.long_context_screen --device cuda --output hybrid_memory_v1_4/artifacts/mechanical_cuda_smoke_2026-08-24.json
python -m hybrid_memory_v1_4.learnability_screen --output hybrid_memory_v1_4/artifacts/learnability.json --checkpoint hybrid_memory_v1_4/artifacts/checkpoints/learned.pt
python -m hybrid_memory_v1_4.validation_screen --output hybrid_memory_v1_4/artifacts/validation.json --checkpoint-dir hybrid_memory_v1_4/artifacts/checkpoints
```

The CUDA screen is a random-token mechanical smoke: finiteness, chunk replay,
bounded cache, and raw throughput only. It is explicitly not model-quality or
matched speed evidence.

## Nonclaims

- The frozen G4a retrieval capability claim failed in both routing cohorts.
- The matched control also failed, so G4a does not isolate selected memory as
  the cause.
- The positive v1.4.4 result uses explicit association/write labels and is not
  evidence of label-free or natural-language learning.
- G8 validates a continuation of exposed G7 checkpoints, not a fresh-from-
  scratch replay of the complete final schedule.
- G9 failed the fresh combined-schedule gate; its 93.93% L512 mean cannot hide
  the 87.89% weak seed.
- The v1.4.5 retention-safe replay used an exposed failed seed and cannot pass
  a fresh-seed gate by itself.
- The retained validation checkpoints are not released pretrained models.
- Straight-through routing establishes a gradient estimator, not successful
  label-free routing.
- The selected-attention topology result establishes a causal gradient path,
  not aligned descent or improved accuracy.
- Spin(8) structure and learned rung use are hypotheses, not promoted
  advantages.
- Reference Python timings are not fused-kernel comparisons.
