# Hybrid Memory v1.4 frontier results

**Date:** 2026-08-24
**Status:** v1.4.4 tied-address successor passed prospectively frozen G4f
fresh-seed validation. G4a through G4e all failed their declared gates. There
is still no label-free, natural-language, or matched-speed promotion.

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

G4b then failed its primary gate:

| Fresh model seed | L96 exact | L512 exact | L512 bits/query |
|---:|---:|---:|---:|
| 1423 | 96.997% | 96.143% | 0.200 |
| 1427 | 96.411% | 94.141% | 0.291 |
| 1429 | 89.087% | 84.814% | 0.801 |

Mean L512 accuracy was 91.699%, but the preregistration required every seed to
reach 90%. The minimum was 84.814%, so the disposition is failure. Before any
validation checkpoint received more training, one identical 300-update L512
consolidation phase for all three seeds was frozen in
[`G4C_PREREGISTRATION.md`](G4C_PREREGISTRATION.md). This tests the optimization-
basin hypothesis without erasing the G4b failure.

Evidence:
[`LEARNABILITY_DIAGNOSIS.md`](LEARNABILITY_DIAGNOSIS.md),
[`artifacts/learnability_g4b_seed1401_cuda_2026-08-24.json`](artifacts/learnability_g4b_seed1401_cuda_2026-08-24.json),
and
[`artifacts/learnability_g4b_long_continuation_seed2401_cuda_2026-08-24.json`](artifacts/learnability_g4b_long_continuation_seed2401_cuda_2026-08-24.json).

Fresh-seed evidence:
[`artifacts/learnability_g4b_validation_cuda_2026-08-24.json`](artifacts/learnability_g4b_validation_cuda_2026-08-24.json).

## G4c failure and v1.4.2 successor

The uniformly frozen 300-update length-512 consolidation improved all seeds
but still failed its all-seed gate:

| Seed | L96 exact | L512 exact |
|---:|---:|---:|
| 1423 | 97.852% | 96.875% |
| 1427 | 97.290% | 96.387% |
| 1429 | 90.784% | 87.598% |

Mean L512 accuracy was 93.620%, but the minimum was 87.598%. G4c is therefore
a failure, not a repaired G4b result.

The post-G4c diagnostic used common fresh cohorts and found:

- every seed had 100% address top-1 accuracy in all four heads;
- removing Gated Delta reduced accuracy to 0.98-3.52%;
- removing attention left accuracy almost unchanged;
- weak seed 1429 had lower query-key margin, higher key cross-correlation,
  larger filler value norm, and larger fast-weight state norm;
- thresholding tiny filler writes helped only modestly.

The remaining failure was therefore content interference/value conditioning,
not router discovery or missing attention. v1.4.2 made three targeted changes:

1. key dimension 16 -> 32 per head;
2. fixed-norm value injection; and
3. retrieval deep supervision immediately after the Gated Delta block.

On already exposed development seeds, including weak seed 1429, v1.4.2
produced:

| Development seed | L96 exact | L512 exact |
|---:|---:|---:|
| 1401 | 97.656% | 97.168% |
| 1429 | 95.129% | 92.676% |

This passed the development falsifier but is not validation. Fresh model seeds
1451/1453/1459 and the requirement that every seed reach 90% at both L96 and
L512 were frozen in [`G4D_PREREGISTRATION.md`](G4D_PREREGISTRATION.md) before
those seeds ran.

Evidence:
[`artifacts/learnability_g4c_consolidation_validation_cuda_2026-08-24.json`](artifacts/learnability_g4c_consolidation_validation_cuda_2026-08-24.json),
[`artifacts/g4c_optimization_diagnostic_cuda_2026-08-24.json`](artifacts/g4c_optimization_diagnostic_cuda_2026-08-24.json),
and
[`artifacts/learnability_v1_4_2_successor_development_cuda_2026-08-24.json`](artifacts/learnability_v1_4_2_successor_development_cuda_2026-08-24.json).

## G4d failure and v1.4.3 identity path

v1.4.2 G4d again failed the strict all-seed gate:

| Fresh seed | L96 exact | L512 exact |
|---:|---:|---:|
| 1451 | 94.543% | 92.090% |
| 1453 | 95.532% | 93.018% |
| 1459 | 91.284% | 86.328% |

Mean L512 accuracy was 90.479%, but the 86.328% minimum blocks promotion. The
interference repair improved the mean without eliminating the random value-
path basin.

v1.4.3 makes the memory read identity-preserving at initialization:

- value projection starts as identity;
- output projection starts as identity;
- the gate starts as `1 + tanh(0) = 1`; and
- the Gated Delta residual starts at sigmoid(0) = 0.5 instead of 0.119.

At the unchanged development budget this raised prior weak seeds 1429 and 1459
to 95.557% and 90.381% at L512. Because 1459 cleared narrowly and still had a
high final-phase loss, G4e prospectively doubles the final 16-pair phase to
1,200 updates. Fresh seeds 1481/1483/1487, 774,400 useful labels per seed, and
the same all-seed 90% gate were frozen in
[`G4E_PREREGISTRATION.md`](G4E_PREREGISTRATION.md).

Evidence:
[`artifacts/learnability_v1_4_2_g4d_validation_cuda_2026-08-24.json`](artifacts/learnability_v1_4_2_g4d_validation_cuda_2026-08-24.json)
and
[`artifacts/learnability_v1_4_3_identity_successor_development_cuda_2026-08-24.json`](artifacts/learnability_v1_4_3_identity_successor_development_cuda_2026-08-24.json).

## G4e failure and v1.4.4 tied address

Doubling the final training phase did not eliminate seed sensitivity. G4e
failed its all-seed gate:

| Fresh seed | L96 exact | L512 exact | L512 bits/query |
|---:|---:|---:|---:|
| 1481 | 79.382% | 78.223% | 0.976 |
| 1483 | 99.280% | 99.316% | 0.045 |
| 1487 | 98.035% | 97.168% | 0.124 |

Mean L512 accuracy was 91.569%, but the minimum was 78.223%. The larger budget
therefore exposed a qualitatively bad basin rather than repairing it.

The post-G4e diagnostic localized the split to address geometry. Seed 1481 had
per-head query-to-write-key top-1 accuracies of 97.66%, 98.83%, 99.80%, and
98.05%, a mean query/key margin of 0.220, and association loss 0.382. Both
strong seeds had 100% address accuracy in every head, margins about 0.41-0.42,
and much lower association loss. Removing Gated Delta reduced all models near
chance; removing attention did not explain the split.

v1.4.4 therefore ties query and key projection weights and initializes that
single address map orthogonally. This is a direct architectural constraint:
queries and writes cannot begin in independently rotated frames. On the
already exposed failed seed 1481, with the exact G4e training budget, it
reached 98.975% at L96 and 97.998% at L512, with 0.078 bits/query. This is a
development repair, not validation.

Fresh seeds 1511/1523/1531, 774,400 useful labels per seed, and the requirement
that every seed reach at least 90% at both L96 and L512 were frozen in
[`G4F_PREREGISTRATION.md`](G4F_PREREGISTRATION.md) before they were run.

Evidence:
[`artifacts/learnability_v1_4_3_g4e_validation_cuda_2026-08-24.json`](artifacts/learnability_v1_4_3_g4e_validation_cuda_2026-08-24.json),
[`artifacts/g4e_optimization_diagnostic_cuda_2026-08-24.json`](artifacts/g4e_optimization_diagnostic_cuda_2026-08-24.json),
and
[`artifacts/learnability_v1_4_4_tied_qk_development_cuda_2026-08-24.json`](artifacts/learnability_v1_4_4_tied_qk_development_cuda_2026-08-24.json).

## G4f pass: robust commissioned learning

G4f started from clean commit `ffc6efd7ce5feebe9a4ed8814e5df9477925a4f3`
and passed the frozen all-seed gate:

| Fresh seed | L96 exact | L512 exact | L512 bits/query |
|---:|---:|---:|---:|
| 1511 | 98.840% | 98.926% | 0.056 |
| 1523 | 98.718% | 98.096% | 0.089 |
| 1531 | 99.304% | 98.975% | 0.057 |

Mean exact query accuracy was 98.954% at L96 and 98.665% at L512. The minima
were 98.718% and 98.096%, both well above the preregistered 90% threshold. Each
L96 row contains 8,192 unseen decisions and each L512 row 2,048. The artifact
records an empty starting Git status, the preregistration hash, all checkpoint
hashes, 116,186 parameters, 774,400 useful query labels per seed, and the RTX
2070 SUPER/PyTorch 2.12.0+cu130 environment.

This validates one precise claim: with tied address geometry, identity value
paths, and explicit association/write/intermediate supervision, v1.4.4 learns
fresh synthetic 16-pair MQAR robustly at the declared budget. It does not show
that the same architecture discovers the rule from final retrieval loss alone
or from natural text.

Evidence:
[`artifacts/learnability_v1_4_4_g4f_validation_cuda_2026-08-24.json`](artifacts/learnability_v1_4_4_g4f_validation_cuda_2026-08-24.json).

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

- `python -m pytest hybrid_memory_v1_4/tests -q`: **193 passed, 4 skipped**.
- `python -m ruff check hybrid_memory_v1_4`: passed.
- `python -m ruff format --check hybrid_memory_v1_4`: passed.
- The four native-suite skips are guarded optional fused FLA/Mamba paths; WSL
  probes separately exercised the installed actual FLA and Mamba environments.
- Frozen G4a executed at 107,552 parameters for the candidate and 112,290 for
  the common-shell DeltaProduct control, a 4.4053% gap under the 5% gate.

Artifact file hashes are recorded in [`ARTIFACTS.sha256`](ARTIFACTS.sha256).

## Nonclaims

- G4a failed retrieval capability in both routing cohorts.
- G4f validates synthetic retrieval only with explicit association,
  write-event, and intermediate labels; label-free retrieval remains open.
- The retained checkpoints are validation artifacts, not released pretrained
  models.
- No natural-language or bits-per-byte claim exists.
- No fused-kernel, Tensor-Core, or matched speed claim exists.
- No learned Spin(8) rung-use claim exists.
- A straight-through estimator is not evidence that label-free routing learns.
