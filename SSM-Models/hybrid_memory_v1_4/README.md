# Hybrid Memory SSM v1.4

**Status:** v1.4.5 remains the model-architecture successor. Its
retention-safe G10 configuration passed the prospectively frozen fresh-seed
gate with minimum exact accuracy 97.57% at L96 and 96.44% at L512. It also
passed the frozen single-seed G11 ordinary next-byte TinyStories screen,
improving from 8.028 to 1.614 bits/byte without auxiliary labels. G12 adds a
validated training recipe without changing the model version: a geometry-aware
Muon/scalar-moment/AdamW composite and lossless 512-token ByteLevel BPE. The
three-seed parameter-matched BPE arm reached 1.534 mean bits per raw byte, but a
post-pretraining factual-recall probe remained negative. G13 then trained an
exact-target `256 -> 512 -> 1,024 -> 2,048 -> 4,096` curriculum. It improved
4,096-token BPRB in all three fresh seeds, but the mean gain was only 0.0126,
short of the frozen 0.02 gate, and factual recall still failed. G9's failed
v1.4.4 seed and the exposed-seed causal repair remain retained evidence. The
frozen G4a selected-memory result remains negative; these are bounded
TinyStories results, not general language-quality or scaling-law promotion.

G14 has since established a narrow edit-law separation: decoupled GDN2 learns
the constructed multi-value accumulation law on all three seeds while tied
GDN-v1 cannot represent it. This is mechanism evidence only. G15 now contains
an experimental content-addressed Spin/Clifford fast-weight memory with exact
identity, fixed `SO(2)^4`, constrained `SU(3)` torus, full Spin(8), and broken-
coupling transports. G15A now passes its exact three-seed SM75 gate: S scores
1.00 on the oracle-coordinate symmetry task versus 0.20 for the fixed torus
and 0.10 for both identity arms, while every arm learns the finite no-symmetry
delayed-value task through length 1,024. This is narrow mechanism evidence,
not generic language promotion, and the v1.4.5 default plan has not changed.
The frozen controls then showed that S ties full Spin with an identity-copy
second read, while beating broken shared coupling by 0.70/0.80/0.80 across the
three seeds. Thus the supported component is the shared vector/positive Spin
lift, not the fixed Clifford/negative-spin read.
See the
[`frontier review`](FRONTIER_REVIEW_2026-08-25.md),
[`Spin/torus research note`](SPIN_TORUS_RESEARCH.md), and
[`G15 result ledger`](G15_SPIN_DIRAC_RESULTS.md).
The now-frozen
[`G15A operational protocol`](G15A_EXECUTION_PROTOCOL_2026-08-25.md) and
[`runner`](g15a_spin_dirac_cohort.py) supply the exact seeds, task generators,
FP32 budget, per-seed promotion semantics, and clean-worktree artifact
requirements that the structural preregistration did not specify. The
[`conditional-controls protocol`](G15A_CONDITIONAL_CONTROLS_PROTOCOL_2026-08-25.md)
and [`conditional artifact`](artifacts/g15a_conditional_controls_sm75_2026-08-25.json)
bind the completed Clifford-read and broken-coupling attribution tests.
The prospective
[`G15A-L learned-coordinate protocol`](G15AL_LEARNED_COORDINATE_PROTOCOL_2026-08-25.md)
then targeted the remaining autonomous controller problem. It failed: S-broken
learned an effective positive-carrier chart matching S within `4e-7` and tied
S on every cosine evaluation. The cosine observation removes the vector-side
alignment scalar, so the shared lift is not identifiable under that scoring
law. The next gate must use full-frame, norm-sensitive observations before
G15B. The prospectively frozen
[`G15A-F four-probe protocol`](G15AF_FULL_FRAME_PROTOCOL_2026-08-25.md)
now does so: every composition is scored through four shared orthogonal frames,
and training is forbidden unless a rank-56 tangent screen and an exhaustive
broken-control Lie-bracket witness pass first. The bound
[`post-hoc observability artifact`](artifacts/g15al_observability_diagnostic_sm75_2026-08-25.json)
records the raw carrier disagreement that G15A-L cosine hid.
The clean G15A-F cohort passed those rank and control certificates and S beat
identity, fixed torus, and S-broken by at least `0.05` in all nine quality
rows. It still failed promotion: S mean relative frame error was
`0.0705--0.1433`, above the frozen `0.05` ceiling in every row. Thus the loss
now sees and separates the shared lift, but accurate autonomous chart learning
remains unsolved. See the
[`G15A-F artifact`](artifacts/g15af_full_frame_cohort_sm75_2026-08-25.json).
The bound
[`support/amplitude decomposition`](artifacts/g15af_learning_diagnostic_sm75_2026-08-25.json)
localizes the miss to off-axis leakage: support-projecting the learned tables
passes every old absolute gate without correcting their amplitudes. The frozen
[`G15A-R first-order protocol`](G15AR_FIRST_ORDER_PROTOCOL_2026-08-25.md)
therefore tests LR decay, a new per-token rotation-covariant second moment, and
a balanced primitive/inverse curriculum separately before fresh confirmation.
G15A-R passed. All five development recipes qualified; even the longer
fixed-LR control passed, so decay is not necessary on current evidence. The
predeclared selectable order chose the unchanged global scalar optimizer on
random compositions with staged LR. On three untouched confirmation seeds, S
mean relative error was `1.21e-7--1.89e-7` through L1,024 and beat I, C, and
S-broken by at least `0.153`. See the
[`G15A-R artifact`](artifacts/g15ar_first_order_repair_sm75_2026-08-25.json).
This solves primitive chart learning under oracle frames/edit timing, not
generic memory or language.

The prospectively frozen
[`G15A-S protocol`](G15AS_SPANNING_CENTER_PROTOCOL_2026-08-25.md) then expanded
the hidden dictionary to both signs of all 28 planes, separated 64 training
from 64 evaluation probe banks, and added direct-carrier assays for global
center words that the two-sided frame cannot distinguish. The clean exact
SM75 run passed every gate on seeds 2281/2287/2293. S held-out-frame mean error
was `6.89e-7--1.18e-6`, its worst structured direct-carrier error was
`2.36e-5`, and the smallest I/C/S-broken mean-error margin was `0.266`.
This supports full-chart and center-compatible composition under oracle edit
timing; it still does not learn addressing, writes, queries, topology from raw
data, generic association, or language. See the
[`G15A-S artifact`](artifacts/g15as_spanning_center_sm75_2026-08-25.json).

The prospectively frozen
[`G15B interleaved-controller protocol`](G15B_CONTROL_PROTOCOL_2026-08-25.md)
and its [`executable cohort`](g15b_interleaved_cohort.py) now target that actual
bottleneck. It removes attention, uses one causal-conv
Spin-Dirac block, interleaves writes/queries/changed overwrites over one shared
payload alphabet, and binds unique-key address, balanced write/erase,
oracle-capacity, controller-identification, and causal-use gates. This is
explicit label-supervised commissioning, not autonomous discovery. The
completed [G15B result](G15B_INTERLEAVED_CONTROLLER_RESULTS.md) fails: identity
learns near-perfect addresses and causally used MQAR/selective memory, but the
token-local controller cannot identify collision-conditioned erase, overwrite
accuracy remains below gate, and Spin transport is inferior to identity on
every non-needle mean cell. G15C and the external-loss-only lane remain
blocked.

Before any fresh retraining, the completed
[`G15B-R0 checkpoint repair`](G15BR_CHECKPOINT_REPAIR_RESULTS.md) replays only
the retained identity checkpoints in their learned address/value gauge. Exact
baseline replay and the temporal-observability witness pass, but soft erase-
equals-write delta correction degrades every non-needle cell. Exact atomic
timing collapses because the models learned a structured one-token write
continuation. The next candidate must anchor erase to every locally observable
write event while allowing a short learned write window; naive tied delta
training is not authorized.

The follow-on
[`G15B-R1 event-erase result`](G15BR1_EVENT_ERASE_RESULTS.md) also fails. It
preserves the learned write continuation bitwise and changes only erase at
every locally observable write event, yet both soft and unit erase lose all
nine non-needle gates. Overwrite falls by 9.7--11.5 points rather than
improving. Mean absolute off-diagonal learned-key cosine is `0.54822`, with a
`0.999934` maximum, so symmetric rank-one erase is not an isolated-slot update
in the retained representation. No event-erase training is authorized.

The completed
[`G15B-R2 collision-only erase result`](G15BR2_COLLISION_ERASE_RESULTS.md)
closes the missing factorial cell. Removing learned erase outside collisions
improves MQAR by 0.8--1.2 points and pre-overwrite queries by 1.8--2.1, but
symmetric erase at a true same-key overwrite makes that stratum 10.3--12.1
points worse. Perfect collision timing and the full learned write tail are
therefore insufficient. The completed
[`G15B-R3 result`](G15BR3_LOGICAL_COMPONENT_RESULTS.md) gives each commissioned
key an oracle replaceable component. It raises ordinary overwrite by
12.2--12.8 points over learned erase and reaches 1.0 in every constructed guard
cell, but fails the frozen gate because the guard baseline is already saturated
and two learned-replay FP32 logit residuals exceed `5e-4` despite identical
predictions. This supports component replacement as a mechanism without
authorizing training. The completed
[`G15B-R4 ownership/background result`](G15BR4_OWNERSHIP_BACKGROUND_RESULTS.md)
then separates value-only from value-plus-`t+1` ownership and query-time
background support. Both value-plus-tail arms pass, including the arm that
excludes background, while both value-only arms fail; seed 2311 collapses
under value-only replacement. The retained checkpoints therefore implement a
two-token write transaction rather than a value-local slot. The frozen
decision remains **do not train** until a causal, non-overlapping transaction
boundary is demonstrated. See the
[`R4 protocol`](G15BR4_OWNERSHIP_BACKGROUND_PROTOCOL_2026-08-26.md) and
[`exact-SM75 artifact`](artifacts/g15br4_ownership_background_sm75_2026-08-26.json).
The completed
[`G15B-R5 causal tail-source result`](G15BR5_CAUSAL_TAIL_SOURCE_RESULTS.md)
separates strict-history, current-token, bias-only, and exact-residual tail
sources. `h_lww_bgminus` passes all 132 frozen performance and bias-separation
checks, reaching 0.9424--0.9466 mean ordinary overwrite and 1.0 on the
constructed guard; current-only and bias-only arms fail. The formal quality
adjudication remains failed because no-reset BPQ replay differs by at most
`1.40e-7` against a frozen `1e-12` bound and FP32 state/read residuals reach
`2.38e-6`/`3.58e-6` against `2e-6`. Discrete replay, learned logits, causal
locality, and FP64 algebra pass. No training is authorized before a separately
frozen R5-S numerical ratification. R5-S has now completed and formally fails:
all 135 source/cell checks exceed its prospective scaled-logit allowance even
though predictions, BPQ, component state/read bounds, transition identity, and
independent FP64 algebra pass. The retained-checkpoint repair route therefore
stops. Preserve R5's history-source evidence, but make the next fresh model's
ordinary residual/read path exact by construction and learn an explicit causal
pending-write/commit state. See the
[`R5 protocol`](G15BR5_CAUSAL_TAIL_SOURCE_PROTOCOL_2026-08-26.md) and
[`exact-SM75 artifact`](artifacts/g15br5_causal_tail_source_sm75_2026-08-26.json),
plus the
[`R5-S result`](G15BR5S_NUMERICAL_RATIFICATION_RESULTS.md) and
[`R5-S artifact`](artifacts/g15br5s_numerical_ratification_sm75_2026-08-26.json).
The fresh architectural pivot is now implemented under the
[`G15B-T transactional-delta protocol`](G15BT_TRANSACTIONAL_DELTA_PROTOCOL_2026-08-26.md).
Its
[`Phase-0 qualification`](G15BT_PHASE0_QUALIFICATION_RESULTS.md) passes every
exact clean-SM75 implementation gate: `F` and `T` each have 38,082 active
parameters and 5,632 batch-2 FP32 state bytes; current-token mutation leaves
the history/edit path bit-identical, prior-history effect is nonzero, maximum
transition spectral norm is `0.9995000000000012`, and FP64/FP32 maximum
logit residuals are `2.78e-16`/`8.94e-8` with exact predictions. All declared
gradient paths are finite and nonzero. The
[`Phase-1 result ledger`](G15BT_PHASE1_RESULTS.md) now records a separate exact
clean-SM75 execution smoke from commit `b3fd297`: all three 67,033-parameter
arms execute their paired schedules, and the strict-history arms pass causal,
chunk, mask-compaction, intervention-reconstruction, and tail-role contracts.
The smoke is non-promotable and is not learned-memory evidence; the frozen
three-seed quality cohort remains pending. R5 and R5-S remain frozen failed
retained-checkpoint results. See the
[`Phase-0 artifact`](artifacts/g15bt_phase0_qualification_sm75_2026-08-26.json)
and
[`Phase-1 smoke artifact`](artifacts/g15bt_phase1_smoke_sm75_2026-08-26.json).

In parallel, the frozen
[`G16 SM75 shootout protocol`](G16_SM75_FRONTIER_SHOOTOUT_PROTOCOL_2026-08-25.md)
and [`harness`](frontier_shootout.py) completed the one-seed, 4.096M-target
development cohort after exact runtime qualification. Official fused Mamba-2
wins every ordinary-compression context and reaches `1.48571` BPRB at L4096
versus v1.4.5 `1.58335`. Local GDN2 (`1.60457`) and actual Transformers OLMo
Hybrid (`1.61413`) lose; every arm fails learned recall. See the
[`G16 result`](G16_SM75_FRONTIER_SHOOTOUT_RESULTS.md). It is parameter- and
target-matched but optimizer-specific and one seed, so it rejects the current
local development candidates without promoting a universal model family.

This track exposes eight explicit mixer kinds in one causal language-model
shell:

- bounded sliding-window RoPE attention with a complete streaming KV cache;
- content-addressed Gated DeltaNet v1 fast-weight memory;
- decoupled GDN2 semantic memory with independent channelwise decay, erase,
  and write;
- matched full-view/strict-history transactional delta with scalar commit,
  symmetric scalar erase, channelwise writes, and exact monolithic readout;
- the maintained repository DeltaProduct reference;
- hierarchical selected-block affine memory;
- bounded rung-routed Spin(8) memory; and
- content-addressed two-sided Spin/Clifford fast weights.

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
- [`gated_delta_v2.py`](gated_delta_v2.py): maintained semantic GDN2 edit law
  with independent channelwise decay, erase, and write; exact scan/chunk/mask/
  gradient contracts; and no fused-kernel claim.
- [`attention.py`](attention.py): bounded causal local attention and cache.
- [`structured_tier.py`](structured_tier.py) and
  [`structured_memory.py`](structured_memory.py): data-routed subgroup ladder
  and a recurrent Spin(8) state that actually changes the model output.
- [`spin_dirac_memory.py`](spin_dirac_memory.py): content-addressed `8_v ->
  8_s+` matrix memory with bounded edits, two-sided transport, exact fixed
  torus arms, full Spin actions, and a Clifford-coupled read.
- [`transactional_delta.py`](transactional_delta.py): fresh matched `F/T`
  transactional fast weight with strict-history edit controls, scalar commit
  and erase, channelwise write, exact scan/chunk/step execution, and no Spin
  transport.
- [`model.py`](model.py): eight-kind hybrid shell, local convolution caches,
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
- [`natural_text_data.py`](natural_text_data.py): fixed-row TinyStories snapshot
  through the Hugging Face Dataset Viewer, with Hub revision, license, byte
  hashes, and exact split-overlap accounting.
- [`natural_text_screen.py`](natural_text_screen.py) and
  [`natural_text_diagnostic.py`](natural_text_diagnostic.py): the frozen G11
  ordinary next-byte comparison against actual Transformers models and its
  post-hoc causal mixer ablation.
- [`tokenization.py`](tokenization.py), [`optimizers.py`](optimizers.py),
  [`natural_text_frontier.py`](natural_text_frontier.py), and
  [`compute_matched_frontier.py`](compute_matched_frontier.py): lossless
  Hugging Face ByteLevel BPE, the geometry-aware composite optimizer, and the
  frozen G12 robustness/allocation experiments.
- [`long_context_curriculum.py`](long_context_curriculum.py) and
  [`long_context_diagnostic.py`](long_context_diagnostic.py): exact-target
  256-to-4,096 training, live-state chunking, recall ablations, and the
  post-hoc positionwise/fast-weight overwrite diagnostic.
- [`g14_gate_law_screen.py`](g14_gate_law_screen.py): frozen constructed
  representational separation between tied GDN-v1 and decoupled GDN2.
- [`native_sm75_probe.py`](native_sm75_probe.py) and
  [`pretrained_sm75_probe.py`](pretrained_sm75_probe.py): fail-closed,
  source-bound native-kernel and actual-checkpoint qualification on the local
  RTX 2070 SUPER. Environment and eligibility details live in
  [`SM75_NATIVE_RUNTIME.md`](SM75_NATIVE_RUNTIME.md).

## Baseline boundary

The normal matched quality control is a `HybridMemoryLM` with the same shell
and a `delta_product` layer plan. A static ProductKey table is not a fair
episodic MQAR control. The FLA DeltaRule registry entries are operator-level
semantic/systems controls, not silently promoted language models. Official
Mamba-2 is an optional separate complete-model comparison and fails closed
when unavailable.

The exact pretrained `state-spaces/mamba2-130m` and
`state-spaces/mamba3-siso-187m` weights are stored outside Git in the WSL Linux
filesystem and loaded through a source-bound official `mamba_ssm` checkout.
Hub revisions, weight hashes, tokenizer provenance, and actual SM75 outcomes
are recorded in [`SM75_NATIVE_RUNTIME.md`](SM75_NATIVE_RUNTIME.md); these
execution probes are not quality comparisons.

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

G10 froze the same retention-safe candidate and complete G9 schedule on unseen
seeds 1753/1759/1777. All passed. L96 mean/minimum were 98.03%/97.57%; L512
mean/minimum were 97.51%/96.44%. In a separate diagnostic, every checkpoint
fell to 0% when Gated Delta was ablated, while removing attention preserved
97.66% accuracy. See [`G10_PREREGISTRATION.md`](G10_PREREGISTRATION.md) and
the retained artifacts.

G11 supplies the first bounded ordinary real-text screen. On one paired
TinyStories seed and 8,192,000 next-byte training labels per model, v1.4.5
improved from 8.028 to 1.614 held-out bits/byte. Actual Transformers Mamba-2
reached 1.639 and actual Transformers OLMo Hybrid reached 1.675. These models
have 119,962, 86,000, and 104,152 parameters respectively, so the ordering is
not a parameter-matched superiority claim.

A post-hoc v1.4.5 ablation found 1.614 bits/byte for the full model, 6.410
without Gated Delta, and 1.818 without attention. The recurrent memory carries
most of the learned real-text function in this checkpoint; attention adds a
smaller measurable improvement. Because the ablation was post-hoc and G11 has
one model seed, cross-seed natural-text robustness remains open.

## G12 optimizer and tokenizer frontier

G12 closes that cross-seed question at the tested scale. The selected
`HarmonicMuonAdamW` optimizer uses PyTorch Muon for hidden matrices, a custom
scalar-second-moment AdamW update for memory controllers, and ordinary AdamW
for embeddings, norms, biases, and convolution. Against raw-byte AdamW with
identical parameters, bytes, and windows, it improves final BPRB on every fresh
seed; the mean moves from 1.8072 to 1.7478 while median update time rises about
4.2%.

The train-only 512-token ByteLevel BPE round-trips both corpus splits exactly
and covers 2.302 training bytes/token. At the closest allowed parameter point
(124,534 versus 119,962 parameters), three-seed mean/worst BPRB are
1.5344/1.5446 and median update time is 79.93 ms. A separately calibrated
measured-CUDA point matches the raw update budget within 4.1% and reaches
1.5498 mean BPRB. These views expose different original-byte counts and must
not be treated as an architecture-only win.

Ordinary validation stays finite at 1,024 tokens, but the frozen templated
counterfactual recall probe does not pass: gains are tiny and seed-inconsistent.
The present bottleneck is missing long-range identification under length-256
ordinary pretraining, not inability to fit text or an unstable state. See
[`OPTIMIZER_TOKENIZER_AUDIT.md`](OPTIMIZER_TOKENIZER_AUDIT.md) for the complete
decision and evidence boundaries.

## G13 exact-target 4,096-token frontier

G13 gives both arms the exact same 4,096 target IDs on every update. The fixed
control partitions each macro-window into sixteen length-256 rows; the
curriculum partitions it into batches `16/8/4/2/1` at lengths
`256/512/1,024/2,048/4,096`. Thus each paired seed receives the same 4,096,000
target tokens and 9,424,359 represented raw bytes. Initial weights, optimizer,
and the entire phase-1 trajectory are also identical.

| Evaluation length | Fixed-256 BPRB | Curriculum BPRB | Curriculum - fixed |
|---:|---:|---:|---:|
| 256 | 1.5929 | 1.5941 | +0.0012 |
| 512 | 1.5916 | 1.5863 | -0.0053 |
| 1,024 | 1.5919 | 1.5827 | -0.0092 |
| 2,048 | 1.5928 | 1.5814 | -0.0115 |
| 4,096 | 1.5932 | 1.5805 | -0.0126 |

The curriculum wins at 4,096 in all three seeds and does not materially harm
256-token loss, but misses the preregistered -0.02 mean requirement. The
ordinary long-context gate therefore fails by magnitude rather than direction.

The stronger memory gate fails decisively. At 8,192 raw bytes, all 72 prompts
were 3,379--3,755 BPE tokens long and therefore exceeded the actual 1,024-token
attention window. The preregistration said 2,048, but the frozen builder and
artifact record 1,024; both paired arms used that same smaller window.
Curriculum matching-minus-mismatched gain averaged only
`0.0000108` nats, versus `0.0000034` for fixed training, with one of three
curriculum seed means negative. Suppressing Gated Delta removes the already
tiny signal, while suppressing attention does not; this is a microscopic
recurrent trace, not robust recall.

The post-hoc causal diagnostic explains the split. Removing Gated Delta raises
curriculum 4,096-token loss by 1.3921 BPRB, whereas removing attention costs
0.0093. The recurrent layer is genuinely useful for ordinary compression.
However, learned mean write strengths are 0.446--0.768 per head per token.
Along the currently written key direction, the measured transition factors are
only 0.553, 0.232, 0.329, and 0.382 despite global retention near 0.999. The
retention floor protects directions that are not rewritten; it does not protect
a one-shot fact from thousands of high-strength content updates in a small
fast-weight matrix.

This keeps v1.4.5 as the model version but changes the next research decision.
Another optimizer, tokenizer, or longer unmodified curriculum is not the
best-supported repair. The next candidate needs a separate slow, sparse-write
memory timescale and an explicitly named self-supervised binding/span objective
derived from natural text. Ordinary next-token loss should remain present as
the language-quality control; commissioned memory training must not be
mislabelled as ordinary pretraining.

## Validation

From `SSM-Models`:

```powershell
python -m pytest hybrid_memory_v1_4/tests -q
python -m hybrid_memory_v1_4.temporal_observability_screen --output hybrid_memory_v1_4/artifacts/temporal_observability_2026-08-24.json
python -m hybrid_memory_v1_4.precision_screen --output hybrid_memory_v1_4/artifacts/precision_horizon_65536_cpu_2026-08-24.json
python -m hybrid_memory_v1_4.long_context_screen --device cuda --output hybrid_memory_v1_4/artifacts/mechanical_cuda_smoke_2026-08-24.json
python -m hybrid_memory_v1_4.learnability_screen --output hybrid_memory_v1_4/artifacts/learnability.json --checkpoint hybrid_memory_v1_4/artifacts/checkpoints/learned.pt
python -m hybrid_memory_v1_4.validation_screen --output hybrid_memory_v1_4/artifacts/validation.json --checkpoint-dir hybrid_memory_v1_4/artifacts/checkpoints
python -m hybrid_memory_v1_4.g14_gate_law_screen --output hybrid_memory_v1_4/artifacts/g14_gate_law_screen_2026-08-25.json
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
- G11 validates bounded ordinary next-byte learning on one TinyStories seed,
  not general language quality; G12 supplies three-seed robustness only at the
  bounded TinyStories scale.
- G12 does not establish robust long-range factual recall, a scaling exponent,
  or hardware-general compute efficiency.
- G13 improves long-context ordinary compression consistently but fails its
  frozen magnitude and factual-recall gates. It is not a long-range-memory
  promotion or a compute-matched win.
- The retained validation checkpoints are not released pretrained models.
- Straight-through routing establishes a gradient estimator, not successful
  label-free routing.
- The selected-attention topology result establishes a causal gradient path,
  not aligned descent or improved accuracy.
- Spin(8) structure and learned rung use are hypotheses, not promoted
  advantages.
- G14 is a constructed unequal-parameter state-law separation, not a language
  or long-context result.
- G15A passes only an oracle-coordinate symmetry mechanism plus a finite
  learned no-symmetry retrieval task. Its controls support the shared
  vector/positive Spin lift but not a contribution from the fixed Clifford
  second read. G15A-L fails autonomous shared-lift attribution because its
  cosine observation admits a compensating broken chart. G15B generic
  association, G15C natural text, and G15D scaling remain open.
  The fixed `su3_torus` mode is not a moving `G2/SU(3)` memory, and
  `SpinDirac` is not a geometric Dirac differential operator.
- Native SM75 probes qualify only the exact recorded implementation/runtime;
  first-call compilation time and finite loss are not matched throughput or
  model quality.
- Reference Python timings are not fused-kernel comparisons.
