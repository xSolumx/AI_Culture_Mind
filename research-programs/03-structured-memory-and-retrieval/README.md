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
- G15A-L's learned-coordinate attribution cohort is a clean-SM75 negative result
  from commit `0c49f64` and fresh seeds 2153/2161/2179. `S` mean cosine ranges
  from 0.9902--0.9957 at L64, 0.9803--0.9913 at L256, and 0.9726--0.9862 at
  L1024. Only seed 2161 at L64 meets the absolute `S` thresholds, and no row
  passes the frozen 0.05 comparator gate. After learned chart
  reparameterization, `S-broken` matches `S` within about `6e-8` across every
  row and its effective positive chart matches within `<4e-7`. Analytically, the
  cosine score cancels the positive scalar `<q,Vk>`, leaving the vector carrier
  unobservable and making the broken signed permutation invertible for `P`.
  Autonomous learned-coordinate attribution therefore fails.
- G15A-F is the prospectively frozen full-frame observability repair, run on
  clean commit `503fa82` with exact SM75 seeds 2203/2207/2213. All three
  pre-training screens have numerical rank 56, condition ratios
  0.286--0.310, and minimum primitive projection residual about 0.598; the exact
  broken-carrier Lie-bracket certificate finds 474 mismatches among 784 pairs,
  with maximum integer residual 4. The quality gate nevertheless fails. `S`
  mean relative Frobenius errors are 0.0927/0.0912/0.0705 at L64,
  0.1201/0.1187/0.1036 at L256, and 0.1304/0.1433/0.1392 at L1024; the p95 gate
  fails in every row and the maximum-error gate is not satisfied cohort-wide.
  `S` does beat `I`, `C`, and `S-broken` by at least 0.05 in every seed/length,
  so the observation repair separates the shared lift, but controller precision
  is not solved. The additional `S-broken >= 2x S` check passes only two rows.
- G15A-R completes the first-order repair on exact SM75 from clean commit
  `eca70f0` (artifact SHA-256 begins `be004dea`). All five development recipes
  qualify. Even the nonselectable 600-step `G-fixed/random` control reaches mean
  errors 0.0155--0.0285, so learning-rate decay is not proven necessary. The
  predeclared least-intervention order selects `G-decay/random`, which reaches
  about `1.03e-7`--`1.85e-7` development mean error: staged decay is vastly
  more precise, while a block moment and balanced curriculum are unnecessary.
  Fresh seeds 2251/2267/2273 pass every frozen `I/C/S/S-broken` row at
  L64/L256/L1024. `S` means span `1.21e-7`--`1.89e-7`, p95 is at most
  `2.68e-7`, maximum error at most `3.27e-7`, margins are 0.234--0.355 versus
  `I`, 0.231--0.351 versus `C`, and 0.153--0.234 versus `S-broken`, with
  broken-2x passing every row.
- G15A-S is the distinct spanning-chart and center-sensitive successor, run on
  exact SM75 from clean commit `4067926` with fresh seeds 2281/2287/2293. Its
  56 signed actions cover all 28 generators, and all training/evaluation banks
  pass numerical rank 56. On unseen banks, `S` mean relative Frobenius errors
  span `6.89e-7`--`1.18e-6` through L1024, p95 is at most `1.47e-6`, and
  maximum error at most `1.94e-6`; margins versus every comparator are
  0.266--0.536 and broken-2x passes every seed/length. Structured direct vector
  and positive errors are at most `2.36e-5`, with frame maximum at
  most `3.54e-5`. The result establishes composition-only transfer of a
  learned spanning signed dictionary to unseen frame banks and global center
  words under oracle edit timing. It does not establish learned topology,
  generic association/address/query learning, negative-spin/Clifford utility,
  language, full triality, or scaling/efficiency.
- G15B completes the prospectively frozen commissioned-controller gate on exact
  SM75 from clean commit `bd5045a`, with three seeds for each `I/C/S` arm,
  4,200 updates and 375,360 scored training decisions per seed/arm. Preflight
  and integrity pass, but adjudication fails. Identity reaches about 0.972 mean
  MQAR and 0.975 selective recall, yet overwrite remains 0.768--0.833. `C` is
  worse; `S` is inferior to identity in every MQAR/overwrite/selective mean cell
  and passes noninferiority only on needle cells. Address top-1 is near perfect,
  while overwrite erase recall remains about 0.5 and write F1 misses its gate.
  The controller sees token/local-convolution features, but the erase target
  depends on collision history: the present learning problem is temporal
  observability, not insufficient Spin transport. G15C/external-only is blocked.
  G15A-S remains a separate passed composition result under oracle edit timing.
- G16 completed its one-seed parameter-matched SM75 development shootout.
  Official fused Mamba-2 wins ordinary compression at every context and beats
  v1.4.5 by `0.09764` BPRB at L4096. Local GDN2 and actual OLMo Hybrid lose;
  all four arms fail learned recall. This does not establish a fresh-seed model
  promotion or scaling law.

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
- G15A's supplied-coordinate primary and conditional mechanism ladders pass,
  but G15A-L rejects autonomous learned-coordinate attribution under its frozen
  cosine observation map. G15A-F restores comparator separation under full-
  frame raw-Frobenius observation but misses the absolute precision gates.
  G15A-R repairs precision with the existing global `ScalarSecondMomentAdamW`,
  600 steps, and staged decay; block-scalar moments and curriculum are not
  needed. G15A-S then passes signed 28-generator composition transfer to unseen
  full-rank frame banks and center words, still under oracle edit timing. G15B
  fails the subsequent commissioned interleaved controller gate and blocks
  G15C/external-only. G15B-R0 rejects naive erase-equals-write delta because the
  learned controllers use a structured one-token write continuation. G15B-R1
  preserves that continuation but rejects erase at every valid write: all nine
  non-needle gates fail, overwrite drops 9.7--11.5 points, and learned-key
  prototype overlap is high. G15B-R2 then preserves the write tail and supplies
  perfect collision timing, but collision-only symmetric erase lowers
  post-same-key-overwrite recall by 10.3--12.1 points. G15B-R3's oracle per-key
  component reset repairs ordinary overwrite by 12.2--12.8 points and reaches
  1.0 on a constructed guard, but fails the frozen saturated-baseline
  improvement and FP32 replay-tolerance gates. G15B-R4 then runs the frozen
  ownership/background factorial with exact R3 replay and `4.44e-15` maximum
  FP64 residual. Only `VT/BG+` and `VT/BG-` pass. Value-only `V/BG+`
  reaches only 0.7030/0.7645/0.7534 ordinary overwrite versus learned
  0.7678/0.8328/0.8288 and collapses on seed 2311; its constructed guard is
  only 0.9390--0.9418. Removing background destroys value-only behavior but
  leaves both value-plus-tail arms near 0.90--0.95 overwrite and 1.0 guard
  accuracy. The useful retained association is therefore localized to
  ambiguous `t+1` ownership, not shared background. G15B-R5 then decomposes
  that tail into strict-history, current-token, bias-only, and exact-residual
  sources without changing the full-token transition. The background-free
  strict-history LWW arm passes every frozen performance and bias-separation
  check: mean overwrite is 0.9424/0.9456/0.9466 and every constructed-guard
  cell is 1.0. The formal exact-SM75 quality adjudication still fails. R4 BPQ
  replay differs by at most `1.40e-7` against the frozen `1e-12` bound, and
  FP32 no-reset-state/background-read residuals reach
  `2.38e-6`/`3.58e-6` against `2e-6`; batch fingerprints, discrete replay,
  learned logits, observability, and FP64 algebra pass. This is a zero-update
  retained-checkpoint mechanism result, not authorization to train. The
  prospectively frozen R5-S cohort then fails every scaled-logit source/cell
  gate (`0/135`) even though categorical behavior, BPQ, state/read bounds,
  bit-exact transitions, fingerprint/provenance checks, and FP64 algebra pass.
  The retained-checkpoint repair route is closed. The next fresh model needs
  explicit pending-write/commit semantics and an exact monolithic residual/read
  path. Generic transport remains identity by default;
  Spin belongs in supplied/coherent-frame
  specialized tasks. Learned topology, full three-carrier triality utility,
  natural-text, and scaling remain open.
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
- G15A-L's high cosines are not a pass: only one seed-length row meets the
  absolute `S` thresholds, no row meets the comparator gate, and `S-broken`
  remains observationally equivalent after learned reparameterization.
- G15A-F's all-row comparator separation is not a quality pass or model
  promotion: `S` misses its frozen absolute mean and p95 gates throughout, and
  the maximum-error and broken-2x checks pass only two rows each.
- G15A-R is a composition-only controller result under a four-probe oracle
  initial frame and oracle edit timing. It is not generic association, learned
  addressing/querying, language modelling, full triality, or scaling evidence;
  the fixed-LR control prevents a claim that decay is necessary.
- G15A-S widens the composition test to a signed 28-generator dictionary,
  unseen frame banks, and center words, but still supplies oracle edit timing.
  It is not topology discovery, generic association, learned address/query
  control, negative-spin/Clifford utility, language modelling, full triality,
  or scaling/efficiency evidence.
- G15B's near-perfect learned address classification is not successful edit-law
  learning. Its overwrite erase recall remains about 0.5, write F1 misses the
  frozen gate, and the quality adjudication fails. The result rejects the tested
  token/local-convolution controller and transport promotion; it does not erase
  G15A-S's separate oracle-timing composition result.

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
- [G15A-L learned-coordinate protocol](../../SSM-Models/hybrid_memory_v1_4/G15AL_LEARNED_COORDINATE_PROTOCOL_2026-08-25.md)
- [G15A-L learned-coordinate artifact](../../SSM-Models/hybrid_memory_v1_4/artifacts/g15al_learned_coordinate_cohort_sm75_2026-08-25.json)
- [G15A-F full-frame protocol](../../SSM-Models/hybrid_memory_v1_4/G15AF_FULL_FRAME_PROTOCOL_2026-08-25.md)
- [G15A-F full-frame artifact](../../SSM-Models/hybrid_memory_v1_4/artifacts/g15af_full_frame_cohort_sm75_2026-08-25.json)
- [G15A-R first-order repair protocol](../../SSM-Models/hybrid_memory_v1_4/G15AR_FIRST_ORDER_PROTOCOL_2026-08-25.md)
- [G15A-R first-order repair artifact](../../SSM-Models/hybrid_memory_v1_4/artifacts/g15ar_first_order_repair_sm75_2026-08-25.json)
- [G15A-S spanning-center protocol](../../SSM-Models/hybrid_memory_v1_4/G15AS_SPANNING_CENTER_PROTOCOL_2026-08-25.md)
- [G15A-S spanning-center artifact](../../SSM-Models/hybrid_memory_v1_4/artifacts/g15as_spanning_center_sm75_2026-08-25.json)
- [G15B interleaved-controller protocol](../../SSM-Models/hybrid_memory_v1_4/G15B_CONTROL_PROTOCOL_2026-08-25.md)
- [G15B interleaved-controller result](../../SSM-Models/hybrid_memory_v1_4/G15B_INTERLEAVED_CONTROLLER_RESULTS.md)
- [G15B exact-SM75 quality artifact](../../SSM-Models/hybrid_memory_v1_4/artifacts/g15b_interleaved_controller_sm75_2026-08-26.json)
- [G15B-R0 checkpoint-repair result](../../SSM-Models/hybrid_memory_v1_4/G15BR_CHECKPOINT_REPAIR_RESULTS.md)
- [G15B-R0 exact-SM75 artifact](../../SSM-Models/hybrid_memory_v1_4/artifacts/g15br_checkpoint_repair_sm75_2026-08-26.json)
- [G15B-R1 event-erase result](../../SSM-Models/hybrid_memory_v1_4/G15BR1_EVENT_ERASE_RESULTS.md)
- [G15B-R1 exact-SM75 artifact](../../SSM-Models/hybrid_memory_v1_4/artifacts/g15br1_event_erase_sm75_2026-08-26.json)
- [G15B-R2 collision-only erase result](../../SSM-Models/hybrid_memory_v1_4/G15BR2_COLLISION_ERASE_RESULTS.md)
- [G15B-R2 exact-SM75 artifact](../../SSM-Models/hybrid_memory_v1_4/artifacts/g15br2_collision_erase_sm75_2026-08-26.json)
- [G15B-R3 component-replacement result](../../SSM-Models/hybrid_memory_v1_4/G15BR3_LOGICAL_COMPONENT_RESULTS.md)
- [G15B-R3 exact-SM75 artifact](../../SSM-Models/hybrid_memory_v1_4/artifacts/g15br3_logical_component_sm75_2026-08-26.json)
- [G15B-R4 ownership/background result](../../SSM-Models/hybrid_memory_v1_4/G15BR4_OWNERSHIP_BACKGROUND_RESULTS.md)
- [G15B-R4 exact-SM75 artifact](../../SSM-Models/hybrid_memory_v1_4/artifacts/g15br4_ownership_background_sm75_2026-08-26.json)
- [G15B-R5 causal-tail-source result](../../SSM-Models/hybrid_memory_v1_4/G15BR5_CAUSAL_TAIL_SOURCE_RESULTS.md)
- [G15B-R5 exact-SM75 artifact](../../SSM-Models/hybrid_memory_v1_4/artifacts/g15br5_causal_tail_source_sm75_2026-08-26.json)
- [G15B-R5-S numerical result](../../SSM-Models/hybrid_memory_v1_4/G15BR5S_NUMERICAL_RATIFICATION_RESULTS.md)
- [G15B-R5-S exact-SM75 artifact](../../SSM-Models/hybrid_memory_v1_4/artifacts/g15br5s_numerical_ratification_sm75_2026-08-26.json)
- [G16 SM75 frontier-shootout protocol](../../SSM-Models/hybrid_memory_v1_4/G16_SM75_FRONTIER_SHOOTOUT_PROTOCOL_2026-08-25.md)
- [G16 trained-frontier results](../../SSM-Models/hybrid_memory_v1_4/G16_SM75_FRONTIER_SHOOTOUT_RESULTS.md)
- [G16 exact-SM75 runtime qualification artifact](../../SSM-Models/hybrid_memory_v1_4/artifacts/g16_runtime_qualification_sm75_2026-08-25.json)
- [Spin/torus architecture and claim boundary](../../SSM-Models/hybrid_memory_v1_4/SPIN_TORUS_RESEARCH.md)

The Spin-labelled reports remain at their provenance paths. Their routing,
update-law, and kernel conclusions are classified here; representation-specific
claims are classified in Programme 04.
