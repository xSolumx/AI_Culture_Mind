# Hybrid Memory v1.4 frontier results

**Date:** 2026-08-24
**Status:** v1.4.1 content-addressed development result awaiting prospectively
frozen fresh-seed validation. The frozen G4a retrieval campaign remains a
capability failure. Retained Gated Delta development checkpoints learned the
synthetic rule under explicit association labels; there is no label-free,
natural-language, or matched-speed promotion.

## Adjudication

The inherited implementation passed its mechanical unit suite, but its
original temporal-observability claim did not survive direct testing. In a
one-block final-only model, reads at non-final positions cannot affect the
loss, and hard coarse `argmax` routes disconnect their logits. This is a
structural counterexample to original gate G3, not a training failure.

The implementation was repaired without relabeling that negative result:

- selected memory now supports dense `soft` and exact-forward
  `straight_through` coarse routing for training;
- sparse `physical_gather` remains hard-only and rejects relaxed routing;
- a `selected_block -> attention` schedule supplies the missing temporal path;
- selected and structured readouts use bounded zero-state conditioning rather
  than division by dtype epsilon;
- experiment artifacts record complete source hashes and exact dirty/untracked
  Git state.

The next scientific gate was the prospectively frozen G4a MQAR campaign in
[`PREREGISTRATION.md`](PREREGISTRATION.md). Both its label-free Q1 cohort and
explicitly supervised Q2 ablation have now run and failed capability.

## Temporal observability

The deterministic float64 audit uses seed 42 and a weighted final-position
objective.

| Case | Coarse routes connected | Non-final read path | Disposition |
|---|---:|---:|---|
| hard, one selected block | no | no | original G3 fails |
| straight-through, one selected block | yes | no | route estimator alone is insufficient |
| straight-through selected block then attention | yes | yes | successor topology accepted |

In the accepted topology, maximum non-final read-logit gradient norms were
`2.1689345e-9` for the coarse block route and `3.8651026e-12` for the fine
route. These are causal gradient-path observations only. They do not establish
aligned descent, routing accuracy, retrieval, or language modeling.

Evidence:
[`artifacts/temporal_observability_2026-08-24.json`](artifacts/temporal_observability_2026-08-24.json).

## Long-context arithmetic

The CPU precision screen completed 65,536 steps without a non-finite value in
the reference, recurrent, or parallel paths.

| Transition | dtype | recurrent max error | parallel max error |
|---|---:|---:|---:|
| generic high-retention affine | fp32 | `9.5783e-5` | `7.4921e-5` |
| generic high-retention affine | fp16 | `1.6465` | `1.5772` |
| selected-block transition | fp32 | `2.6410e-7` | `2.6259e-7` |
| selected-block transition | fp16 | `2.0505e-3` | `4.0036e-3` |
| maintained DeltaProduct transition | fp32 | `3.0126e-8` | `2.7387e-8` |
| maintained DeltaProduct transition | fp16 | `1.8806e-4` | `3.1405e-4` |

The large fp16 error in the deliberately high-retention generic stress case is
retained rather than averaged away. This artifact is numerical error-growth
evidence, not a trained-model result.

Evidence:
[`artifacts/precision_horizon_65536_cpu_2026-08-24.json`](artifacts/precision_horizon_65536_cpu_2026-08-24.json).

## CUDA execution

On the Windows PyTorch 2.12.0+cu130 runtime and RTX 2070 SUPER, a 30,801-
parameter `selected_block -> attention` model produced finite logits and state,
replayed chunks exactly under the screen's acceptance check, and held actual
and capacity cache size at 33,032 bytes through lengths 512, 2,048, and 8,192.
Raw observed rates were 1,183.23, 1,192.07, and 1,207.75 tokens/s.

This is a random-token mechanical smoke on an unfused Python reference. It is
not a matched throughput comparison or a quality measurement.

Evidence:
[`artifacts/mechanical_cuda_smoke_2026-08-24.json`](artifacts/mechanical_cuda_smoke_2026-08-24.json).

## Retrieval harness smoke

A two-update CUDA run exercised fresh paired MQAR episodes, pre/post
evaluation, optimizer updates, straight-through routing, complete chunk state,
and atomic artifact writing. The smoke configuration used one seed, lengths
32/64, and much smaller models than G4a. Both rows remained at zero exact
accuracy after two updates.

That zero is neither a negative capability result nor evidence for the model:
the artifact is explicitly non-evidentiary and exists only to test the
training/evaluation plumbing.

Evidence:
[`artifacts/retrieval_training_smoke_cuda_2026-08-24.json`](artifacts/retrieval_training_smoke_cuda_2026-08-24.json).

## Frozen G4a retrieval campaign

The evidentiary campaign used the preregistered three seeds, 600 updates,
length-512 training, paired fresh MQAR episodes, and exact evaluation at
lengths 512, 2,048, and 8,192. Pre-training exact accuracy was 0% in every row.
The parameter gate passed at 107,552 candidate versus 112,290 control
parameters, a 4.4053% gap.

Mean post-training exact per-query accuracy was:

| Cohort and model | L512 | L2,048 | L8,192 |
|---|---:|---:|---:|
| Q1 label-free selected-attention | 2.214% | 1.563% | 1.563% |
| Q1 DeltaProduct control | 2.344% | 1.823% | 1.172% |
| Q2 supervised selected-attention | 1.693% | 2.474% | 2.083% |
| Q2 DeltaProduct control | 2.344% | 1.823% | 1.172% |

Every candidate seed missed the frozen greater-than-90% L512 capability gate
in both cohorts. Q1 candidate-minus-control mean difference at L2,048 was
`-0.2604` percentage points; Q2's was `+0.6510` points. Both avoided a paired
regression worse than two points, but neither can be promoted because the
capability prerequisite failed. The repeated control rows replayed exactly
across Q1 and Q2, as expected because routing supervision does not act on that
model.

This result falsifies the frozen G4a capability claim at the declared budget.
It does not isolate selected memory as the cause because the common-shell
DeltaProduct control also remained near chance. Explicit route supervision did
not rescue the candidate. A successor campaign needs a known-capable complete-
model positive control and a separately frozen optimization diagnostic before
another architectural comparison.

Evidence:
[`artifacts/retrieval_g4a_q1_label_free_cuda_2026-08-24.json`](artifacts/retrieval_g4a_q1_label_free_cuda_2026-08-24.json)
and
[`artifacts/retrieval_g4a_q2_supervised_routing_cuda_2026-08-24.json`](artifacts/retrieval_g4a_q2_supervised_routing_cuda_2026-08-24.json).

## G4b development diagnosis and pivot

The selected candidate stored values but no key signatures. A perfect static
key-mod-slot oracle therefore measured only 66.41% exact accuracy with 16
slots in the development cohort; 32 slots reached 89.06%, and 64 slots reached
100%. The exact values move with the cohort, but a 16-address value-only head
cannot satisfy the frozen greater-than-90% gate.

G4a also presented only 19,200 scored query labels. v1.4.1 replaced the primary
selected tier with content-addressed Gated Delta matrix state and used a
2/4/8/16-pair dense-query curriculum with 467,200 useful labels. The fixed
16-pair batch overfit reached 100%, proving the training shell was live.

Development results were:

| Cohort | Exact query accuracy | Bits/query |
|---|---:|---:|
| fresh 16-pair, L96, 1,024 queries | 94.336% | 0.319 |
| fresh 16-pair, L512, 256 queries | 89.844% | 0.560 |
| disjoint L512 pre-continuation, 2,048 queries | 92.676% | 0.412 |
| same cohort after 300 fresh continuation updates | 94.189% | 0.289 |

The small post-continuation extrapolation cohorts reached 91.41% at L1,024 and
96.88% at L2,048, each over 128 queries. They are too small for a long-context
promotion.

The association auxiliary explicitly aligns query vectors with matching write
keys, labels write events, and discourages filler decay. The result is
therefore a successful label-supervised commissioning result, not label-free
learning. Fresh model seeds 1423/1427/1429 and the requirement that every seed
reach at least 90% over 2,048 unseen L512 queries were frozen in
[`G4B_PREREGISTRATION.md`](G4B_PREREGISTRATION.md) before those seeds ran.

Evidence:
[`LEARNABILITY_DIAGNOSIS.md`](LEARNABILITY_DIAGNOSIS.md),
[`artifacts/learnability_g4b_seed1401_cuda_2026-08-24.json`](artifacts/learnability_g4b_seed1401_cuda_2026-08-24.json),
and
[`artifacts/learnability_g4b_long_continuation_seed2401_cuda_2026-08-24.json`](artifacts/learnability_g4b_long_continuation_seed2401_cuda_2026-08-24.json).

## Actual upstream probes

- FlashRT Gated Delta Attention was pinned at revision `892f725c...`, kernel
  version 6. Its artifact requires BF16 x86_64 Linux and SM80+, so it correctly
  fails closed on the local SM75 RTX 2070 SUPER.
- Actual FLA 0.5.2 `chunk_gated_delta_rule` produced finite outputs and final
  state in the WSL FLA environment.
- Actual Transformers 5.9.0 `Mamba2ForCausalLM` and
  `OlmoHybridForCausalLM` produced finite CUDA logits through their declared
  unfused fallbacks.
- The actual 128,989,632-parameter pretrained
  `state-spaces/mamba2-130m` checkpoint at revision `3a5aea0c...` was downloaded
  to the E: large cache and produced finite FP16 CUDA logits through official
  `mamba_ssm` in WSL.

These are implementation-availability probes across different runtimes, not a
matched speed ranking or an MQAR quality comparison.

Evidence:
[`artifacts/upstream_fla_gated_delta_wsl_cuda_2026-08-24.json`](artifacts/upstream_fla_gated_delta_wsl_cuda_2026-08-24.json),
[`artifacts/upstream_native_transformers_cuda_2026-08-24.json`](artifacts/upstream_native_transformers_cuda_2026-08-24.json),
and
[`artifacts/upstream_pretrained_mamba2_wsl_cuda_2026-08-24.json`](artifacts/upstream_pretrained_mamba2_wsl_cuda_2026-08-24.json).

## Validation record

- `python -m pytest hybrid_memory_v1_4/tests -q`: **189 passed, 4 skipped**.
- `python -m ruff check hybrid_memory_v1_4`: passed.
- `python -m ruff format --check hybrid_memory_v1_4`: passed.
- The four native-suite skips are guarded optional fused FLA/Mamba paths; WSL
  probes separately exercised the installed actual FLA and Mamba environments.
- Frozen G4a executed at 107,552 parameters for the candidate and 112,290 for
  the common-shell DeltaProduct control, a 4.4053% gap under the 5% gate.

Artifact file hashes are recorded in [`ARTIFACTS.sha256`](ARTIFACTS.sha256).

## Nonclaims

- G4a failed retrieval capability in both routing cohorts.
- G4b development learned synthetic retrieval only with explicit association
  and write-event labels; label-free retrieval remains open.
- The retained checkpoints are development artifacts, not released pretrained
  models.
- No natural-language or bits-per-byte claim exists.
- No fused-kernel, Tensor-Core, or matched speed claim exists.
- No learned Spin(8) rung-use claim exists.
- A straight-through estimator is not evidence that label-free routing learns.
