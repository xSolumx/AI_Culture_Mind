# Programme 03: Learned structured memory and retrieval systems

**Research author:** Hayden Austin

**Last reconciled:** 2026-08-25

## Scope

Memory-core quality and systems behaviour for direct slots, delta memories,
hard and soft routing, hierarchical large-slot layouts, physically gathered
state, and eager or fused kernels. This programme asks how addresses, update
laws, state layout, and kernels interact; it does not assume an exceptional
representation.

## Core Questions

1. Which failures come from address geometry rather than the memory update?
2. Can coarse-to-fine routing preserve or improve retrieval while touching
   only selected recurrent state?
3. When do direct, delta, gathered, and fused implementations compute matched
   recurrences, and which systems trade-offs survive controlled measurement?

## Proven / Established Results

- In the frozen 10-seed overwrite campaign, hard/discretized routing is more
  robust than the maintained learned continuous-delta and additive-fast-weight
  rows in every corrupted cell. Oracle delta is exact, so the supported
  conclusion concerns address geometry, not deficient delta capacity.
- Direct and triality-coded slots agree on every clean hard-route cell when
  router and transport are matched. Soft-route differences are small and
  sign-unstable.
- A same-router two-slot hierarchy improves mean cosine for both direct and
  delta memories in the frozen synthetic campaign; hard routing makes the two
  update laws exactly equal on that separable world.
- The transported co-moving implementation includes prefix, solve, and
  read-frame costs. On the recorded RTX 2070 SUPER at length 4,096 it is about
  `5.20x` faster forward and `5.30x` faster forward plus backward than the
  direct transported reference, while using 128 rather than 64 logical state
  scalars.
- In the later 64-slot overlapping-semantic campaign, shared routing and
  hierarchy pass all ten frozen seeds. Actual eager gathering preserves the
  recurrence but loses through launch overhead; the separately frozen fused
  gathered-state inference kernel beats the named eager dense and gathered
  controls across the tested grid.
- The reported direct, native-delta, gathered, and fused rows are tied to
  complete structured artifacts and explicit hardware/protocol boundaries.
- In the Spin-Delta two-key grammar, a causal width-three hard router with
  auxiliary labels identifies all write/query events and slots perfectly over
  three seeds and 8/16/32-write evaluation. Joint end-to-end retrieval still
  fails its robust gate, localizing the remaining defect to co-adaptation or
  recurrent optimization rather than final address classification.

### Current hybrid-memory boundary

- Hybrid Memory v1.4.5 is a validated small causal learner and a validated
  commissioned synthetic associative-memory model. Its successful default
  shell is gated-delta memory followed by bounded attention; this does not
  establish the older structured Spin(8) cache as the source of the gain.
- G11 reached 1.614 held-out bits/byte in a bounded one-seed TinyStories
  next-byte screen, versus 1.639 for the actual Mamba-2 comparator and 1.675
  for the OLMo hybrid. The models were not parameter matched, so this is a
  learning validation rather than a general quality ranking.
- G12 supports bounded multi-seed ordinary-text improvements under its exact
  tokenizer/optimizer protocol but does not validate factual recall; tokenizer
  arms also consumed unequal raw-byte exposure.
- G13's exact-target 4,096-token curriculum improved all paired ordinary-loss
  rows but missed its frozen effect-size threshold, and 8,192-byte factual
  recall remained tiny and sign-unstable. Long-context archive promotion
  therefore failed.
- G14 shows, on a deliberately constructed unequal-parameter mechanism task,
  that independent channel-wise erase/write can represent accumulation that a
  tied scalar erase/write gate cannot. It prioritizes GDN2 but is not a model-
  quality or natural-text result.
- G15A completed its exact SM75/FP32 primary cohort over three fresh seeds.
  Full Spin `S` reached 1.0 symmetry macro versus 0.2 for commuting `C` and 0.1
  for both identity `I` and fixed-Clifford-read `I+C` in every seed. Every arm
  also learned the no-sym delayed-recall control at 1.0 through L1024. The
  prospectively frozen conditional controls are also complete:
  `S+identity-read` ties `S` at 1.0 symmetry macro in all seeds, so a fixed
  Clifford/negative-spin read contribution is not supported, while `S-broken`
  scores 0.3/0.2/0.2 versus `S` at 1.0, passing the shared-coupling control with
  0.7/0.8/0.8 margins. Both conditional arms retain 1.0 no-sym delayed recall
  through L1024. Because the task supplies exact coordinates and oracle carrier
  controls, the strongest result is shared vector/positive-Spin lift on this
  designed task, not full three-carrier triality, generic association,
  natural-text recall, or scaling.

## Open Claims

- End-to-end training with a recent window, selected fine blocks, and a
  compressed global summary has not been completed.
- The selected-block mechanism is not yet registered as an FLA/Hugging Face
  mixer with recurrent-cache and backward contracts.
- The fused gathered kernel is inference-only; gradient/training kernels,
  cross-GPU thresholds, and production integration remain open.
- The action frame and physical block layout are supplied in the large-slot
  campaign. Joint frame/layout discovery remains open.
- Comparisons with current fused Gated DeltaNet/DeltaProduct, p-BIM,
  Householder-product, and sparse-attention systems remain necessary.
- A matched natural-text and delayed-binding comparison of GDN-v1, KDA-style
  channel decay, and GDN2 remains the immediate update-law gate after G14.
- G15A's primary and prospectively frozen conditional mechanism ladders pass.
  Full three-carrier triality, autonomous coordinate inference, generic
  association, natural-text, and scaling gates remain later work.
- A frozen phase-separated follow-up rejected early address noise as a
  complete explanation: the perfect frozen-router core still missed its robust
  gate in one seed.
- A fresh exact-control 3x3 factorial then detected both initialization and
  minibatch-order sensitivity, including sign-changing interactions. The open
  mechanism is core optimization geometry, not final event/slot inference.
- A repository-wide mechanism audit ranks short-to-long write-depth curriculum
  as the next falsifier, ahead of local reconstruction credit and an
  action-coordinate optimizer audit. It explicitly rejects more routing,
  slots, exceptional transport, or compiler tuning as the immediate repair.
- The resulting exact-control paired 3x3 curriculum gate passed every frozen
  repair condition. Its worst 16-write cell rose from 93.51% to 99.37%, and
  maximum factorial sensitivity contracted from 6.49 to 0.63 points despite
  24.52% fewer training tokens. Transfer through the learned causal router is
  still open. The first transfer cohort failed its bitwise router-replication
  gate before quality summarization; a fresh single-router-execution repair is
  prospectively frozen. That v2 repair passed: the autonomous learned-router
  curriculum raised the worst 16-write cell from 78.96% to 97.02% and reduced
  the largest factorial range from 21.00 to 2.83 points. Removing router-label
  supervision remains open; a retrieval-only joint curriculum gate is now
  prospectively frozen with slot identities scored modulo their global
  permutation symmetry. That gate failed: the curriculum averaged 34.30% at
  16 writes versus 62.84% at fixed depth, and query-event F1 stayed zero even
  in a 98.19%-accurate retrieval cell. The redundant internal-query fallback
  makes the explicit router factorization unidentifiable from final retrieval
  loss alone. A prospectively frozen one-step audit then reproduced the exact
  mechanism: the initialized hard query event leaves its own gradient nonzero
  while multiplying the query-slot gradient by zero. Both isolated repairs
  restored slot credit, but soft event continuation perturbed initial logits
  `2,840x`--`4,076x` less than immediate authority across all three paired
  seeds. The resulting fresh 3x3 training intervention nevertheless failed:
  mean 16-write retrieval fell 9.08 points, its worst regression was 52.88
  points, and minimum query-event F1 remained zero. Query-slot identification
  did improve, so gradient restoration is real but insufficient. The next gate
  is temporal observability of the desired controller under final-only loss,
  not another continuation schedule. That audit proved exact non-final zeros
  for one block and nonzero indirect paths through a second block, but the
  resulting descent directions were near chance with respect to the desired
  event grammar. The explicit binary query event is therefore not an
  identifiable latent factor under this objective.

## Dependencies

- Programme 01 supplies ordered scan and co-moving compiler machinery.
- Programme 02 may supply a shared routing/action prior; Programme 04 supplies
  Spin(8)/Spin(9) instantiations. Neither is required for the direct/delta
  memory law or the fused gathered-state result.
- Controlled end-to-end model evidence remains in the supporting benchmark
  track, not in this memory-core programme.

## Non-claims

- A local kernel speedup is not cross-hardware or production superiority.
- Fused gathered inference is not a training or language-model result.
- Hard-routing success is not a theorem that direct memory is universally
  better than delta memory.
- No matched experiment shows triality- or Spin(9)-specific storage capacity.
- G15A's supplied-coordinate symmetry and shared-coupling wins do not establish
  autonomous transport inference, full three-carrier triality, generic
  association, natural-text recall, or scaling.

## Canonical Evidence

- [Detailed cross-era memory evidence ledger](EVIDENCE_LEDGER.md)
- [Matched learned-retrieval campaign](../../Spin-Space-Research/docs/experiments/MATCHED_LEARNED_RETRIEVAL_RESULTS.md)
- [Scanner optimization and winner-by-regime audit](../../Spin-Space-Research/docs/experiments/SCHURSCAN_MEMORY_SCANNER_OPTIMIZATION_RESULTS.md)
- [Hierarchical memory and transported FLA result](../../Spin-Space-Research/docs/experiments/SPIN8_SPIN9_HIERARCHICAL_MEMORY_RESULTS.md)
- [Large-slot semantic hierarchy and fused gather](../../Spin-Space-Research/docs/experiments/LARGE_SLOT_SEMANTIC_HIERARCHY_RESULTS.md)
- [Memory benchmark atlas and FLA hybrid-model fit](../../Spin-Space-Research/docs/experiments/MEMORY_BENCHMARK_ATLAS.md)
- [Historical Task B replay provenance failure](../../Spin-Space-Research/docs/experiments/TASK_B_DELTA_ACTION_REPLAY_RESULTS.md)
- [Spin-Delta mechanism reuse audit](../../SSM-Models/pure_spin_ssm_v1_2/SPIN_DELTA_MECHANISM_REUSE_AUDIT.md)
- [Spin-Delta write-curriculum result](../../SSM-Models/pure_spin_ssm_v1_2/SPIN_DELTA_WRITE_CURRICULUM_RESULTS.md)
- [Learned-router curriculum-transfer result](../../SSM-Models/pure_spin_ssm_v1_2/SPIN_DELTA_ROUTER_CURRICULUM_TRANSFER_RESULTS.md)
- [Label-free curriculum result](../../SSM-Models/pure_spin_ssm_v1_2/SPIN_DELTA_LABEL_FREE_CURRICULUM_RESULTS.md)
- [Query gradient-topology result](../../SSM-Models/pure_spin_ssm_v1_2/SPIN_DELTA_QUERY_GRADIENT_TOPOLOGY_RESULTS.md)
- [Query-event continuation result](../../SSM-Models/pure_spin_ssm_v1_2/SPIN_DELTA_QUERY_CONTINUATION_RESULTS.md)
- [Temporal query observability result](../../SSM-Models/pure_spin_ssm_v1_2/SPIN_DELTA_TEMPORAL_OBSERVABILITY_RESULTS.md)
- [Hybrid v1.4/v1.4.5 complete result ledger](../../SSM-Models/hybrid_memory_v1_4/RESULTS.md)
- [G14 decoupled edit-law preregistration](../../SSM-Models/hybrid_memory_v1_4/G14_PREREGISTRATION.md)
- [G15 Spin-Dirac status and result ledger](../../SSM-Models/hybrid_memory_v1_4/G15_SPIN_DIRAC_RESULTS.md)
- [G15A primary cohort artifact](../../SSM-Models/hybrid_memory_v1_4/artifacts/g15a_spin_dirac_cohort_sm75_2026-08-25.json)
- [G15A conditional attribution protocol](../../SSM-Models/hybrid_memory_v1_4/G15A_CONDITIONAL_CONTROLS_PROTOCOL_2026-08-25.md)
- [G15A conditional attribution artifact](../../SSM-Models/hybrid_memory_v1_4/artifacts/g15a_conditional_controls_sm75_2026-08-25.json)
- [Spin/torus architecture and claim boundary](../../SSM-Models/hybrid_memory_v1_4/SPIN_TORUS_RESEARCH.md)

The Spin-labelled reports remain at their provenance paths. Their routing,
update-law, and kernel conclusions are classified here; representation-specific
claims are classified in Programme 04.
