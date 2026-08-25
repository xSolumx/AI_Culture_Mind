# Hybrid Memory v1.4 frontier results

**Date:** 2026-08-25
**Status:** v1.4.5 remains the validated retention-safe successor. G10 passed
fresh synthetic seeds, G11/G12 established bounded ordinary TinyStories
learning, and G13 found a consistent but sub-threshold 4,096-token compression
gain with negative factual recall. G14 passed only its constructed decoupled-
edit mechanism gate. G15A now passes a three-seed oracle-coordinate mechanism
and finite learned no-symmetry gate, but not generic association or natural
text. There is still no robust long-range recall, broad language-quality,
scaling-law, or matched-speed promotion.

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

## G5 actual upstream learning comparison

G5 paired the same model seed, episode stream, optimizer, and 2,200-update
curriculum across v1.4.4 and the actual Transformers Mamba-2 and OLMo Hybrid
implementations. Parameter counts were 116,186, 82,224, and 100,376. Both
upstream models explicitly used their native unfused PyTorch fallbacks.

| Model | L96 exact | L512 exact | L512 bits/query |
|---|---:|---:|---:|
| Hybrid Memory v1.4.4 | 5.688% | 4.443% | 5.763 |
| Transformers Mamba-2 | 17.969% | 14.160% | 4.563 |
| Transformers OLMo Hybrid | 97.522% | 25.439% | 9.003 |

This does not contradict G4f. G4f used explicit internal association,
write-event, and intermediate retrieval supervision; G5 withheld all internal
state labels. OLMo Hybrid learned the trained L96 distribution but did not
retain that performance across the longer filler span. The local model did not
autonomously discover its address/write controls.

G5 also diagnosed an error in its intended external auxiliary. Predicting the
random value from the preceding write key is statistically impossible because
that value has not appeared and is sampled independently. Every model's
reconstruction loss stayed near `ln(64)`. The term was shared fairly but added
noise rather than binding information.

G5b therefore freezes a corrected external target: at the already observed
value event, reconstruct the preceding key. This is causal, learnable, and
available to every architecture without inspecting internal memory. Fresh
seed 1621 and the exact successor protocol are frozen in
[`G5B_PREREGISTRATION.md`](G5B_PREREGISTRATION.md).

Evidence:
[`artifacts/g5_actual_upstream_learning_comparison_cuda_2026-08-24.json`](artifacts/g5_actual_upstream_learning_comparison_cuda_2026-08-24.json).

## G5b causal reverse-binding result

The corrected auxiliary was learned by every architecture and changed the
outcome materially:

| Model | L96 exact | L512 exact | L512 bits/query |
|---|---:|---:|---:|
| Hybrid Memory v1.4.4 | 97.375% | 95.996% | 0.194 |
| Transformers Mamba-2 | 98.657% | 77.979% | 1.222 |
| Transformers OLMo Hybrid | 100.000% | 58.105% | 3.072 |

All final-phase reverse-binding losses were below 0.004, whereas G5's unseen-
value losses stayed near `ln(64)`. The repaired signal therefore trained an
observable binding rather than injecting random gradients. OLMo and Mamba-2
slightly exceeded v1.4.4 at the trained L96, but v1.4.4 retained much more of
the rule across the L512 filler extension.

This is a clean-start, preregistered, paired comparison using real upstream
library models, but it is still only model seed 1621. It supports a mechanism
hypothesis, not a superiority claim. Fresh v1.4.4 model seeds 1643/1657/1663
and an all-seed 90% L96/L512 gate are frozen in
[`G6_PREREGISTRATION.md`](G6_PREREGISTRATION.md).

Evidence:
[`artifacts/g5b_causal_reverse_binding_comparison_cuda_2026-08-24.json`](artifacts/g5b_causal_reverse_binding_comparison_cuda_2026-08-24.json).

## G6 failure and competence-paced successor

The three-seed external-learning gate did not pass:

| Fresh seed | L96 exact | L512 exact | L512 bits/query |
|---:|---:|---:|---:|
| 1643 | 89.209% | 83.398% | 0.832 |
| 1657 | 96.216% | 94.873% | 0.254 |
| 1663 | 98.694% | 98.193% | 0.086 |

Mean L512 accuracy was 92.155%, but seed 1643 failed both gates. G6 therefore
does not validate robust external-label learning.

The reverse-binding loss was low for every seed, including 0.0058 in weak seed
1643. Its phase-ending retrieval accuracies, however, were only 12.5%, 17.2%,
53.9%, and 87.1%. The fixed schedule advanced based on elapsed updates rather
than acquired competence. Gated Delta ablation reduced the three diagnostic
cohorts to 0-2.15%, while attention ablation left them almost unchanged, so the
retrieval mechanism is the recurrent memory rather than the attention shell.

The older privileged internal query/key top-1 diagnostic no longer tracks
performance under external training: strong seed 1663 scored poorly on that
metric while retrieving well. It is retained as a non-aligned diagnostic, not
used to choose another internal auxiliary.

G7 keeps the architecture and causal objective fixed but changes curriculum
control. Every phase must reach two consecutive 90% fresh competence probes
before advancing, with preregistered per-phase caps. Fresh seeds 1693/1697/1699
and the final all-seed gate are frozen in
[`G7_PREREGISTRATION.md`](G7_PREREGISTRATION.md).

Evidence:
[`artifacts/g6_external_reverse_binding_validation_cuda_2026-08-25.json`](artifacts/g6_external_reverse_binding_validation_cuda_2026-08-25.json)
and
[`artifacts/g6_optimization_diagnostic_cuda_2026-08-25.json`](artifacts/g6_optimization_diagnostic_cuda_2026-08-25.json).

## G7 failure and target-distance successor

Competence pacing did not pass its gate. It did make all seeds strong on the
trained L96, but not on L512:

| Fresh seed | Updates used | L96 exact | L512 exact |
|---:|---:|---:|---:|
| 1693 | 4,400 | 94.312% | 85.889% |
| 1697 | 2,800 | 96.497% | 93.018% |
| 1699 | 5,100 | 96.472% | 89.307% |

Mean L512 accuracy was 89.404%, below the gate. Seeds 1693 and 1699 also hit
the P2 and P4 caps without two consecutive competence probes, then later
mastered P8 and P16. The assumption that easier-phase retrieval competence is
a monotonic prerequisite was false.

The remaining common failure is distance distribution: all seeds learned L96,
but two did not retain the binding robustly through L512 filler. G8 therefore
restores each saved G7 optimizer and applies the same 600-update, batch-16,
learning-rate-0.001 L512 phase to all seeds. This tests direct target-distance
training without seed-specific tuning. G7 remains failed.

Evidence:
[`artifacts/g7_competence_paced_validation_cuda_2026-08-25.json`](artifacts/g7_competence_paced_validation_cuda_2026-08-25.json).

## G8 pass: explicit target-distance retention

The uniform L512 continuation passed its all-seed gate:

| G7 source seed | L96 exact | L512 exact | L512 bits/query |
|---:|---:|---:|---:|
| 1693 | 93.689% | 92.285% | 0.386 |
| 1697 | 96.704% | 95.801% | 0.223 |
| 1699 | 96.484% | 95.752% | 0.201 |

Mean L96 accuracy was 95.626% with a 93.689% minimum. Mean L512 accuracy was
94.613% with a 92.285% minimum. No seed forgot the short task, and every seed
cleared the frozen 90% gate at both lengths.

G8 began from clean commit `c8aa2326924e335627210e7b325c6eb05824b799`,
restored the exact G7 model and AdamW optimizer states, changed the learning
rate uniformly to 0.001, and supplied 600 batch-16 L512 updates to every seed.
That adds only 38,400 target-distance retrieval labels and 153,600 external
reverse-binding labels per seed.

The present synthetic learning solution is therefore a coordinated schedule:

1. tied query/key geometry removes the independent address-frame lottery;
2. reverse-key reconstruction after the observed value supplies a causal,
   learnable binding/event signal without reading internal memory state;
3. query loss trains use of the stored binding; and
4. a direct L512 phase calibrates retention at the deployment distance instead
   of treating short-length mastery as evidence of extrapolation.

This is a prospectively frozen successful continuation across three failed-G7
checkpoints. It does not retroactively pass G7 and is not a fresh-from-scratch
validation of the entire combined schedule.

Evidence:
[`artifacts/g8_target_distance_consolidation_cuda_2026-08-25.json`](artifacts/g8_target_distance_consolidation_cuda_2026-08-25.json).

G9 prospectively freezes a fresh-from-scratch replay of the complete schedule.
Its fixed phase counts are 1,200/1,200/1,400/1,300 followed by the 600-update
L512 phase. Fresh seeds 1721/1723/1733 must each exceed 90% at both lengths;
the G8 continuation result is unchanged whatever G9 reports.

## G9 failure: learned global decay is the unstable degree of freedom

The fixed fresh-from-scratch schedule failed its all-seed gate:

| Fresh seed | L96 exact | L512 exact |
|---:|---:|---:|
| 1721 | 99.146% | 98.535% |
| 1723 | 88.269% | 87.891% |
| 1733 | 96.106% | 95.361% |

Mean accuracy was 94.507% at L96 and 93.929% at L512, but the seed-1723
minimum failed both declared thresholds. This is not repaired by reporting the
mean. The run began at clean preregistration commit `c0c05b2` and used
1,292,800 retrieval plus 1,408,000 external reverse-binding labels per seed.

The post-failure diagnostic isolates a concrete architectural instability.
For seed 1723, one of four Gated Delta heads learned mean filler write strength
0.9976 and mean filler retention 0.9004, effectively the configured hard floor
of 0.90. Across roughly 416 filler transitions, that global retention factor
alone removes effectively all old-state contribution. The strong seeds kept
the corresponding filler retentions near one.

The memory layer is nevertheless causally used: seed-1723 accuracy was 86.52%
in the diagnostic cohort, fell to 1.95% when Gated Delta was ablated, and
remained 85.35% when attention was ablated. Retrieval gradients into the
memory parameters were finite and nonzero. Reverse-binding loss also converged
for the weak seed. Thus the current learning problem is not disconnected
credit, an unused memory layer, or failure to learn the external association
target. It is a seed-dependent optimizer route in which unrestricted global
decay erases episodic bindings on filler tokens.

The next causal intervention changes only the retention range while holding
architecture, initialization seed, data, optimizer, schedule, and evaluation
fixed. Natural-text expansion remains blocked until that intervention passes
an unseen-seed gate.

Evidence:
[`artifacts/g9_fresh_combined_validation_cuda_2026-08-25.json`](artifacts/g9_fresh_combined_validation_cuda_2026-08-25.json)
and
[`artifacts/g9_optimization_diagnostic_cuda_2026-08-25.json`](artifacts/g9_optimization_diagnostic_cuda_2026-08-25.json).

## v1.4.5 development: retention-only intervention repairs the exposed seed

The matched seed-1723 replay changed only the learned retention range:
minimum/initial retention moved from 0.90/0.995 to 0.999/0.9995. Tensor shapes,
tied address geometry, model seed, data namespaces, objective, schedule,
optimizer, and evaluation cohorts were identical to G9. The hard floor now has
a 692.8-token half-life instead of 6.58 tokens; content-selective delta
overwrites remain trainable.

| Exposed seed 1723 | v1.4.4 G9 | v1.4.5 retention-safe |
|---|---:|---:|
| L96 exact | 88.269% | 98.059% |
| L512 exact | 87.891% | 96.289% |

The phase losses stayed finite and the candidate cleared both development
thresholds. A separate L512 diagnostic cohort reached 98.242%. Disabling the
Gated Delta mixer collapsed accuracy to 0%; disabling attention left 98.242%,
so this was not an attention workaround. The active write head retained
0.99973 on filler rather than the failed model's 0.9004. Its filler write
strength was 0.5592 rather than 0.9976.

This supports the erasure diagnosis as a matched causal result, but seed 1723
was selected because it failed. The candidate remains development-only until
it passes a prospectively frozen unseen-seed cohort.

G10 froze the identical candidate and complete schedule on unseen seeds
1753/1759/1777. Its all-seed 90% gate at L96 and L512 was declared before any
of those checkpoints were trained.

Evidence:
[`artifacts/g9_retention_safe_exposed_seed_cuda_2026-08-25.json`](artifacts/g9_retention_safe_exposed_seed_cuda_2026-08-25.json)
and
[`artifacts/g9_retention_safe_diagnostic_cuda_2026-08-25.json`](artifacts/g9_retention_safe_diagnostic_cuda_2026-08-25.json).

## G10 pass: retention-safe v1.4.5 is robust on fresh seeds

The prospectively frozen cohort passed every declared threshold:

| Fresh seed | L96 exact | L512 exact |
|---:|---:|---:|
| 1753 | 98.022% | 97.900% |
| 1759 | 98.499% | 98.193% |
| 1777 | 97.571% | 96.436% |

Mean/minimum accuracy was 98.031%/97.571% at L96 and 97.510%/96.436% at
L512. G10 began from clean commit `3ea7406b59fffacce3facd0445bd797ceb0bebad`.
Each seed received the exact G9 objective and budget: 5,700 updates, 1,292,800
retrieval labels, and 1,408,000 external reverse-binding labels. No internal
memory labels or seed-specific schedule were used.

The post-gate diagnostic also passed the causal-use check. All three models
scored 97.461%--97.656% on its L512 cohort, and all fell to 0% when the Gated
Delta mixer was disabled. Removing attention left 97.656% for every seed. The
successful function is therefore carried by the retention-safe memory layer,
not by the small attention block.

This promotes v1.4.5 as the synthetic external-causal-learning successor. It
does not convert the reverse-binding auxiliary into ordinary next-token
learning evidence. A separately frozen real-text screen is required for that
question.

The real-text input is now snapshotted before that screen: TinyStories Hub
revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, train rows 0--1999
(1,730,239 UTF-8 bytes), and validation rows 0--255 (230,722 bytes). The
official splits have zero exact-story overlap in the selected prefixes. The
snapshot is retained rather than refetched during training.

Evidence:
[`artifacts/g10_retention_safe_validation_cuda_2026-08-25.json`](artifacts/g10_retention_safe_validation_cuda_2026-08-25.json)
and
[`artifacts/g10_retention_safe_diagnostic_cuda_2026-08-25.json`](artifacts/g10_retention_safe_diagnostic_cuda_2026-08-25.json).

## G11 pass: ordinary real-text next-token learning

G11 removed every synthetic auxiliary and trained only causal UTF-8 next-byte
cross entropy on the retained TinyStories snapshot. Every model saw the same
8,192,000 scored training bytes and fixed 131,072-byte validation cohort.

| Model | Parameters | Initial BPC | 500 | 1000 | Final BPC | Final byte accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Hybrid Memory v1.4.5 | 119,962 | 8.028 | 2.106 | 1.837 | **1.614** | 65.398% |
| Transformers Mamba-2 | 86,000 | 8.394 | 1.845 | 1.718 | 1.639 | 65.193% |
| Transformers OLMo Hybrid | 104,152 | 8.063 | 2.101 | 1.872 | 1.675 | 64.297% |

v1.4.5 improved by 6.414 bits/byte, passing the preregistered requirement of
at least 2.0 improvement and final BPC at most 4.0. All curves were finite. The
actual upstream runtime classes were
`transformers.models.mamba2.modeling_mamba2.Mamba2ForCausalLM` and
`transformers.models.olmo_hybrid.modeling_olmo_hybrid.OlmoHybridForCausalLM`.
Both declared their Torch fallbacks; no repository substitute was used.

The final BPC ordering is descriptive, not a superiority claim: this is one
model seed and the parameter counts differ. Raw wall time is also not a speed
comparison because the upstream models used different unfused paths.

### Post-hoc causal-use diagnostic

The trained hybrid is genuinely using its recurrent memory:

| Frozen v1.4.5 mode | Validation BPC | Byte accuracy |
|---|---:|---:|
| Full | 1.614 | 65.398% |
| Gated Delta disabled | 6.410 | 16.696% |
| Attention disabled | 1.818 | 60.749% |

Removing memory costs 4.796 bits/byte; removing attention costs 0.204. The
learned residual weights were 0.560 for memory and 0.146 for attention. Across
32,768 inspected validation bytes, all four heads actively wrote and mean
retention per head remained near 0.99902. This diagnostic is post-hoc, so it
supports causal use of the frozen checkpoint without becoming a preregistered
architecture comparison.

G11 therefore resolves the narrow learning question positively: v1.4.5 can
learn ordinary real-text causal prediction, and its recurrent memory carries
most of that learned function in this bounded run. General language quality,
cross-seed robustness, scaling, and long-context natural-text recall remain
open.

Evidence:
[`artifacts/g11_tinystories_next_byte_comparison_cuda_2026-08-25.json`](artifacts/g11_tinystories_next_byte_comparison_cuda_2026-08-25.json)
and
[`artifacts/g11_tinystories_memory_ablation_cuda_2026-08-25.json`](artifacts/g11_tinystories_memory_ablation_cuda_2026-08-25.json).

## G12: optimizer, tokenizer, robustness, and matched allocation

The repo-wide audit selected a composite optimizer rather than a blanket
replacement. The exact Spin(8)/SO(8) chart experiment had already shown that
coordinatewise AdamW moments break an orthogonal parameter relation that SGD
preserves. G12 therefore sends 102,400 hidden-matrix parameters to actual
PyTorch Muon, 522 write/decay/residual controller parameters to a custom
scalar-second-moment AdamW update, and all remaining parameters to ordinary
AdamW with semantic no-decay groups.

The corrected paired development screen used identical initial weights,
training windows, validation windows, and presented bytes. Composite minus
AdamW final BPRB was -0.1205 and -0.0948 on seeds 1823 and 1829, so it passed
the frozen -0.02 mean rule. Median update time increased about 3.9%.

The tokenizer audit trained only on the pinned TinyStories training stream:

| Tokenizer | Vocabulary | Train bytes/token | Validation bytes/token | Exact round trip |
|---|---:|---:|---:|---|
| raw UTF-8 bytes | 256 | 1.000 | 1.000 | yes |
| ByteLevel BPE | 512 | 2.302 | 2.337 | yes |
| ByteLevel BPE | 1,024 | 3.038 | 3.048 | yes |

The frozen rule selected the smallest fitted vocabulary exceeding 2.0 training
bytes/token: 512. It has no unknown or special token. Cross-tokenizer loss is
total token negative log likelihood divided by represented original bytes and
`ln(2)`, reported as bits per raw byte.

### Fresh three-seed result

| Arm | Parameters | Mean BPRB | Worst BPRB | Median update | Presented raw bytes |
|---|---:|---:|---:|---:|---:|
| raw + AdamW | 119,962 | 1.8072 | 1.8204 | 108.96 ms | 4,096,000 |
| raw + composite | 119,962 | 1.7478 | 1.7537 | 113.55 ms | 4,096,000 |
| parameter-matched BPE + composite | 124,534 | **1.5344** | **1.5446** | **79.93 ms** | 9,434,111 mean |

The raw optimizer arm wins every paired seed by 0.0442--0.0671 BPRB. The BPE
arm passes the frozen robustness gate, but sees about 2.30 times as many
original bytes for the same 4,096,000 token targets. Its 3.81% parameter
mismatch is the fixed closest-shape result.

### Long context after ordinary pretraining

All checkpoints execute finite ordinary validation at 256, 512, and 1,024
tokens. Mean BPRB changes as follows:

| Arm | 256 tokens | 512 tokens | 1,024 tokens |
|---|---:|---:|---:|
| raw + AdamW | 1.8175 | 1.8363 | 1.8734 |
| raw + composite | 1.7542 | 1.7693 | 1.7516 |
| parameter-matched BPE + composite | 1.5505 | 1.5176 | 1.5075 |

This is longer-context execution and ordinary loss, not recall. The paired
counterfactual factual probe at raw-byte distances 128/256/512/1,024 is
negative: gains are tiny, sign-changing across seeds, and non-monotone. The
BPE row's all-positive 1,024-byte seed means average only 0.0033 nats and are
not promoted as a capability.

### Measured-compute point

G12E enumerated widths 24--96 and FFN expansions 1--6 using random tokens only.
Five warmups and 15 synchronized updates per valid shape selected BPE width 64
/ expansion 1. Calibration was 110.72 ms versus 106.80 ms for the raw AdamW
target (+3.67%). On the three outcome seeds it measured 113.43 ms versus
108.96 ms (+4.10%), with 111,770 parameters, 1.5498 mean BPRB, and 1.5625
worst-seed BPRB. It passes the CUDA-matched Pareto rule versus raw AdamW.

The parameter-matched width-48/expansion-5 BPE point is still faster and
slightly better. Thus extra recurrent width was not the best allocation at
this scale; a narrower recurrent core plus larger FFN dominates among the
tested BPE shapes. This is an RTX 2070 SUPER result, not a scaling exponent.

Complete interpretation is in
[`OPTIMIZER_TOKENIZER_AUDIT.md`](OPTIMIZER_TOKENIZER_AUDIT.md). Evidence:
[`artifacts/g12a_tokenizer_audit_2026-08-25.json`](artifacts/g12a_tokenizer_audit_2026-08-25.json),
[`artifacts/g12b_optimizer_development_cuda_2026-08-25.json`](artifacts/g12b_optimizer_development_cuda_2026-08-25.json),
[`artifacts/g12c_multiseed_natural_text_cuda_2026-08-25.json`](artifacts/g12c_multiseed_natural_text_cuda_2026-08-25.json),
[`artifacts/g12d_post_pretraining_long_context_recall_cuda_2026-08-25.json`](artifacts/g12d_post_pretraining_long_context_recall_cuda_2026-08-25.json), and
[`artifacts/g12e_compute_matched_frontier_cuda_2026-08-25.json`](artifacts/g12e_compute_matched_frontier_cuda_2026-08-25.json).

## G13: exact-target context curriculum through 4,096

G13 asks whether the G12 recipe failed recall simply because it never trained
beyond 256 tokens. The comparison is stronger than a token-count match. For
every seed and update, both arms score the exact same contiguous 4,096 target
IDs and represented raw bytes. Fixed training splits them into sixteen
length-256 rows. The curriculum splits the same macro-window into rows of
length 256, 512, 1,024, 2,048, and 4,096 over five 200-update phases. Recurrent
state remains live across 1,024-token execution chunks and is never detached.

Protocol deviation: the preregistration specified a 2,048-token attention
window, while the frozen builder and completed artifact used 1,024. Both arms
used 1,024, so paired deltas remain internally controlled; G13 did not test the
written 2,048-window architecture.

Fresh seeds 2011/2017/2027 each received 4,096,000 BPE targets and exactly
9,424,359 represented raw bytes per arm. Phase 1 is identical by construction:
all three paired BPRB differences are exactly zero, and final target hashes and
raw-byte counts match for every pair.

| Held-out context | Fixed-256 mean BPRB | Curriculum mean BPRB | Paired mean delta | Curriculum wins |
|---:|---:|---:|---:|---:|
| 256 | 1.5929 | 1.5941 | +0.0012 | 1/3 |
| 512 | 1.5916 | 1.5863 | -0.0053 | 3/3 |
| 1,024 | 1.5919 | 1.5827 | -0.0092 | 3/3 |
| 2,048 | 1.5928 | 1.5814 | -0.0115 | 3/3 |
| 4,096 | 1.5932 | 1.5805 | -0.0126 | 3/3 |

The effect is consistent but smaller than preregistered. The ordinary gate
required at least -0.02 mean BPRB at 4,096; the measured -0.0126 fails that
threshold. The curriculum raises 256-token mean BPRB by only 0.0012, well
inside its guardrail. Positionwise replay shows that the 4,096 gain is genuinely
history-dependent: paired deltas are -0.0021 for positions 1--256, -0.0115 for
257--512, and about -0.0133 thereafter.

This target match is not compute matching. Mean synchronized training time was
92.49 seconds for fixed training and 110.76 seconds for the curriculum. Median
curriculum update time rose from 91.19 ms at length 256 to 127.93 ms at length
4,096. Maximum training allocation was 1,265.1 versus 1,517.9 MiB on the RTX
2070 SUPER.

### Recall remains negative

The full-model counterfactual gains remain too small to promote:

| Raw-byte distance | Fixed mean gain (nats) | Curriculum mean gain (nats) |
|---:|---:|---:|
| 128 | +0.006449 | -0.002048 |
| 256 | +0.009928 | +0.003079 |
| 512 | +0.008960 | +0.008119 |
| 1,024 | +0.004410 | +0.003247 |
| 2,048 | +0.001247 | +0.001666 |
| 4,096 | +0.000088 | +0.000227 |
| 8,192 | +0.000003 | +0.000011 |

At 8,192 raw bytes, prompt lengths are 3,379--3,755 BPE tokens; all 72
full-model rows exceed the actual 1,024-token attention window. Only two of three
curriculum seed means are positive. The mean curriculum-control improvement is
`0.00000734` nats, not the required 0.01, and the absolute mean is not the
required 0.02. The learned-recall and full G13 promotion gates fail.

### Why ordinary memory works while one-shot recall fails

The frozen-checkpoint diagnostic establishes two facts simultaneously:

| Arm | Full BPRB | Gated Delta off | Attention off | Gated Delta cost | Attention cost |
|---|---:|---:|---:|---:|---:|
| Fixed 256 | 1.5932 | 2.9531 | 1.5976 | +1.3599 | +0.0045 |
| Curriculum | 1.5805 | 2.9726 | 1.5898 | +1.3921 | +0.0093 |

The recurrent core carries most ordinary 4,096-token compression; the gain is
not merely longer local attention. But its per-token mean write strengths after
curriculum training are `[0.446, 0.768, 0.671, 0.618]`. Mean retention remains
near 0.999, while the exact measured transition factors along each token's
written key direction are only `[0.553, 0.232, 0.329, 0.382]`. Fixed training
writes even harder, with factors `[0.442, 0.156, 0.204, 0.294]`.

This resolves the apparent contradiction. Global decay is stable, and the
fast-weight state learns useful ordinary-text statistics, but every token can
strongly erase the direction it writes. A finite low-rank content-addressed
matrix optimized only for next-token likelihood is not identified as a
protected archive for rare one-shot bindings. Longer context reduces write
aggression and improves compression, but does not create sparse admission,
protected consolidation, or a query-aligned factual objective.

The next architectural route is therefore a distinct slow sparse-write memory
timescale, trained with an explicitly named self-supervised natural-text
binding/span target alongside ordinary next-token loss. That would be
commissioned memory training, not evidence that ordinary pretraining alone
learned recall. Merely extending the same curriculum, changing tokenizer, or
changing optimizer again is not the best-supported causal intervention.

Evidence:
[`G13_PREREGISTRATION.md`](G13_PREREGISTRATION.md),
[`artifacts/g13_exact_target_long_context_curriculum_cuda_2026-08-25.json`](artifacts/g13_exact_target_long_context_curriculum_cuda_2026-08-25.json), and
[`artifacts/g13_posthoc_long_context_diagnostic_cuda_2026-08-25.json`](artifacts/g13_posthoc_long_context_diagnostic_cuda_2026-08-25.json).

## G14: decoupled edit-law mechanism gate

G14 was frozen before its run and asks a deliberately narrow question: can one
address retain eight independent nonnegative value features when erase and
write are decoupled? Tied GDN-v1 cannot represent the target; its final
coefficient mass is at most one, giving a length-eight MSE lower bound of
`49/64 = 0.765625`. GDN2 can use near-zero erase and near-unit write.

| Arm | Trainable parameters | Three-seed held-out MSE | Bit accuracy |
|---|---:|---:|---:|
| tied GDN-v1 | 10 | 0.770998--0.771060 | 0% |
| decoupled GDN2 | 90 | 0.000181--0.000292 | 100% |

Every frozen G14 gate passed. The result establishes a constructed
representational and learnability separation for the edit law. It is
intentionally unequal-parameter, uses an abstract controller rather than the
language-model shell, and supplies no natural-text, long-context, or model-
quality promotion.

Evidence:
[`G14_PREREGISTRATION.md`](G14_PREREGISTRATION.md),
[`g14_gate_law_screen.py`](g14_gate_law_screen.py), and
[`artifacts/g14_gate_law_screen_2026-08-25.json`](artifacts/g14_gate_law_screen_2026-08-25.json).

## G15: Spin-Dirac implementation and G15A result

The new candidate stores one
content-addressed `8 x 8` association matrix per head, uses independent
erase/write controls, transports it on the left and right through shared
Spin(8) carriers, and optionally applies the fixed Clifford tensor at readout.
It is a 64-state LTV SSM per head and an associative two-sided scan; it is not
the old 24-scalar cache, a scalar Mamba-2 SSD, or a geometric Dirac operator.

Contract tests now pass for recurrent/parallel/chunk/token replay, masked
state, complete LM gradients, center signatures, Clifford equivariance,
bounded transitions, a 4,096-step float64 state falsifier, the fixed
`SO(2)^4` torus, the exact constrained `SU(3)` rank-two torus, and a broken-
coupling control. The primary edit gates were prospectively repaired from
channelwise to independent head-scalar controls after the conjugation audit
showed that diagonal/Hadamard channel gates choose a preferred basis. The
scalar law passes exact shared-frame covariance; channelwise gating remains an
explicit non-equivariant ablation.

The combined preregistered integrity artifact now passes. On SM75 FP32, both
4,096-step heads remain finite and below `9.0e-5` of the analytic ceiling.
Scalar-moment and SGD mapped covariance residuals stay below `1.87e-13`, and
both the delayed coordinate path and scored-position query path clear the
frozen read-change/descent thresholds in all three seeds. This authorizes the
I, I+C, C, and S learning cohort; it did not count as that learning result.
The later prospective
[`G15A execution protocol`](G15A_EXECUTION_PROTOCOL_2026-08-25.md) freezes the
previously absent exact seeds, task support, FP32 budget, optimizer,
per-seed aggregation, clean-worktree rule, and artifact schema before any
runner output was inspected.

The clean quality run at commit `73df687f` subsequently passed all frozen
G15A conditions on seeds 2131, 2137, and 2141. In every seed, S achieved 1.00
macro accuracy on the ten-class supplied-coordinate symmetry task, versus
0.20 for C and 0.10 for I and I+C. Every arm achieved 1.00 learned no-symmetry
accuracy at lengths 64, 256, and 1,024. All arms had 11,508 trainable
parameters, the same parameter-shape and schedule hashes, and 256 bytes of
FP32 recurrent state per sequence. Maximum trained inner-conjugation residual
was below `5.45e-15`; the oracle semantic ladder also passed.

This establishes a strong finite mechanism separation between full Spin
transport and a single fixed torus/identity when exact coordinates and carrier
controls are supplied. It does not establish learned geometry or a natural-
text advantage. The required `S+identity-read` and `S-broken` controls were
frozen prospectively in
[`G15A_CONDITIONAL_CONTROLS_PROTOCOL_2026-08-25.md`](G15A_CONDITIONAL_CONTROLS_PROTOCOL_2026-08-25.md)
and then executed from clean commit `5fc3d7b`. S+identity-read tied S at 1.00
in every seed, so the fixed Clifford/negative-spin read contribution is not
supported. S-broken reached only 0.30/0.20/0.20, giving S margins of
0.70/0.80/0.80 and passing the frozen shared-coupling control. Both controls
retained 1.00 no-symmetry accuracy through length 1,024.

The supported interpretation is a shared vector/positive Spin lift on the
designed supplied-coordinate task, not usefulness of all three triality
carriers. The intentionally broken arm's `0.071--0.078` conjugation residual
is a diagnostic confirming the break, not an integrity failure. The first
execution exposed and preserved a runner adjudication mismatch; the code was
corrected to the already frozen diagnostic-only rule before the evidentiary
rerun, without changing any experimental setting.

Evidence:
[`G15_SPIN_DIRAC_PREREGISTRATION.md`](G15_SPIN_DIRAC_PREREGISTRATION.md),
[`G15_SPIN_DIRAC_AMENDMENT_2026-08-25.md`](G15_SPIN_DIRAC_AMENDMENT_2026-08-25.md),
[`G15_SPIN_DIRAC_EDIT_LAW_AMENDMENT_2026-08-25.md`](G15_SPIN_DIRAC_EDIT_LAW_AMENDMENT_2026-08-25.md),
[`G15A_EXECUTION_PROTOCOL_2026-08-25.md`](G15A_EXECUTION_PROTOCOL_2026-08-25.md),
[`G15A_CONDITIONAL_CONTROLS_PROTOCOL_2026-08-25.md`](G15A_CONDITIONAL_CONTROLS_PROTOCOL_2026-08-25.md),
[`G15AL_LEARNED_COORDINATE_PROTOCOL_2026-08-25.md`](G15AL_LEARNED_COORDINATE_PROTOCOL_2026-08-25.md),
[`G15_SPIN_DIRAC_RESULTS.md`](G15_SPIN_DIRAC_RESULTS.md), and
[`SPIN_TORUS_RESEARCH.md`](SPIN_TORUS_RESEARCH.md), plus
[`artifacts/g15_integrity_sm75_2026-08-25.json`](artifacts/g15_integrity_sm75_2026-08-25.json)
and
[`artifacts/g15a_spin_dirac_cohort_sm75_2026-08-25.json`](artifacts/g15a_spin_dirac_cohort_sm75_2026-08-25.json).
The conditional evidence is
[`artifacts/g15a_conditional_controls_sm75_2026-08-25.json`](artifacts/g15a_conditional_controls_sm75_2026-08-25.json).

G15A-L then learned only the token-to-action coordinate table from delayed
positive-read loss, keeping edit controls and the transported final query
oracle-fixed. It failed the frozen gate. S mean cosine ranged from 0.9902 to
0.9957 at L64, 0.9803 to 0.9913 at L256, and 0.9726 to 0.9862 at L1,024; only
one seed/length cleared the absolute S thresholds. S-broken matched S to
approximately `6e-8` or better on every row.

This is an observation-law failure. The scored positive read has form
`r^(L-1) <q,Vk> Pv`; cosine normalization discards the positive vector-side
alignment scalar. S-broken's invertible signed permutation can therefore
learn an effective positive chart matching S within `4e-7`, which the retained
raw tables verify. The next controlled gate must use multiple independent
queries and unnormalized Frobenius loss before more optimizer tuning or G15B.

Evidence:
[`artifacts/g15al_learned_coordinate_cohort_sm75_2026-08-25.json`](artifacts/g15al_learned_coordinate_cohort_sm75_2026-08-25.json).

## Actual upstream probes

- FlashRT Gated Delta Attention remains ineligible on SM75 because its published
  artifact requires SM80+. It is not replaced by a local approximation.
- Source-current official Mamba-3 SISO at revision `e9594ce1...` passed native
  training forward/backward on SM75 with complete finite input and all-
  parameter gradients. The MIMO TileLang route selected an SM80 TF32 MMA path
  and is excluded.
- Source-current FLA GDN2 recurrent mode produced finite output and input
  gradient, but gradients were absent for all recurrent-core projections. Its
  chunk forward completed, while backward exceeded the bounded 1,200-second
  attempt. Neither route is an eligible SM75 training baseline.
- The source-built `ssiu/flash-attention-turing` extension passed FP16 causal
  and noncausal forward/backward numerical gates at head dimensions 64 and
  128. Official FlashAttention mainline remains Ampere-or-newer; the dedicated
  Turing fork is the actual local implementation.
- The actual 186,849,600-parameter `state-spaces/mamba3-siso-187m` checkpoint
  at Hub revision `6792c27c...` produced finite FP16 logits and loss through a
  source-bound official Mamba checkout. The inaccessible gated Meta tokenizer
  repository is not hidden: the public `NousResearch/Meta-Llama-3.1-8B`
  tokenizer mirror and its exact revision/hashes are recorded.
- The actual 128,989,632-parameter `state-spaces/mamba2-130m` checkpoint at
  revision `3a5aea0c...` also passed after the current causal-conv1d source
  dependency was built for SM75. Its weight and GPT-NeoX tokenizer revisions/
  hashes are bound in the artifact.

These are implementation-availability probes across different runtimes, not
a matched speed ranking or quality comparison. Full package, source,
checkpoint, tokenizer, and exclusion details are in
[`SM75_NATIVE_RUNTIME.md`](SM75_NATIVE_RUNTIME.md).

Evidence:
[`artifacts/native_sm75_mamba3_siso_2026-08-25.json`](artifacts/native_sm75_mamba3_siso_2026-08-25.json),
[`artifacts/native_sm75_gdn2_recurrent_2026-08-25.json`](artifacts/native_sm75_gdn2_recurrent_2026-08-25.json),
[`artifacts/native_sm75_gdn2_chunk_2026-08-25.json`](artifacts/native_sm75_gdn2_chunk_2026-08-25.json),
[`artifacts/native_sm75_flash_turing_2026-08-25.json`](artifacts/native_sm75_flash_turing_2026-08-25.json),
[`artifacts/pretrained_sm75_mamba3_siso_187m_2026-08-25.json`](artifacts/pretrained_sm75_mamba3_siso_187m_2026-08-25.json), and
[`artifacts/pretrained_sm75_mamba2_130m_2026-08-25.json`](artifacts/pretrained_sm75_mamba2_130m_2026-08-25.json).

## Validation record

- `python -m pytest hybrid_memory_v1_4/tests -q`: **261 passed, 4 skipped**.
- `python -m ruff check hybrid_memory_v1_4`: passed.
- `python -m ruff format --check hybrid_memory_v1_4`: passed.
- The four native-suite skips are guarded optional fused operator paths; the
  source-bound WSL probes separately qualify or reject the actual SM75
  implementations.
- Frozen G4a executed at 107,552 parameters for the candidate and 112,290 for
  the common-shell DeltaProduct control, a 4.4053% gap under the 5% gate.

Artifact file hashes are recorded in [`ARTIFACTS.sha256`](ARTIFACTS.sha256).

## Nonclaims

- G4a failed retrieval capability in both routing cohorts.
- G4f validates synthetic retrieval only with explicit association,
  write-event, and intermediate labels; label-free retrieval remains open.
- G8 validates external causal-label target-distance continuation, not ordinary
  label-free next-token learning or a fresh combined-schedule cohort.
- G9 failed its fresh combined-schedule all-seed gate.
- The retention-safe v1.4.5 result is an exposed-seed development intervention,
  not fresh validation.
- G10 is a fresh synthetic external-label validation, not natural-text
  next-token evidence.
- G11 is a one-seed bounded next-byte screen, not general language quality or a
  parameter-matched upstream superiority result. G12 supplies three-seed local
  robustness, not larger-corpus or larger-scale validation.
- G12's post-pretraining counterfactual factual-recall result is negative.
- G12 parameter and CUDA matches are distinct, and neither estimates a scaling
  law or hardware-general efficiency.
- G13 failed its frozen effect-size and factual-recall gates.
- G14 is an unequal-parameter constructed state-law result, not model quality.
- G15A's passing symmetry task supplies exact coordinates and oracle carrier
  controls. Its conditional result supports shared vector/positive Spin
  coupling but not the fixed Clifford second read; it does not establish
  autonomous geometry. G15A-L's learned chart fails because the cosine
  observation cannot identify the vector carrier. No generic association,
  natural text, or moving `G2/SU(3)` memory follows.
- The retained checkpoints are validation artifacts, not released pretrained
  models.
- The natural-text claims are bounded G11--G13 TinyStories results only.
- The passing Turing FlashAttention and Mamba-3 SISO execution gates are
  native-runtime qualifications, not a matched speed or quality result.
- No learned Spin(8) rung-use claim exists.
- A straight-through estimator is not evidence that label-free routing learns.
