# Programme 03: Learned structured memory and retrieval systems

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
  prospectively frozen.

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

The Spin-labelled reports remain at their provenance paths. Their routing,
update-law, and kernel conclusions are classified here; representation-specific
claims are classified in Programme 04.
