# Hierarchical matched-retrieval preregistration

- **Date frozen:** 2026-08-10, before implementation-result inspection
- **Programme:** Triality memory and Intertwiner SchurScans
- **Status:** prospective routing ablation

## Question

Does a parameter-free hierarchy applied to the already trained categorical
router improve overwrite/recall, corruption robustness, and selected-memory
traffic without changing the recurrent-state or encoder-parameter budget?

This is an NSA-inspired routing experiment, not an implementation of Native
Sparse Attention.  It tests the transferable claim that a coarse selector can
restrict fine memory access.  It does not import NSA's token-level KV cache,
attention operator, or fused kernels.

## Frozen policies and budgets

For every seed, train the existing `learned_both_joint` write/query router for
300 optimizer steps at each of the three maintained curriculum radii.  Both
encoders are `8 x 24` matrices, or 384 learned scalars in total.  Every memory
row has eight slots or an `8 x 8` fast-weight matrix: 64 recurrent scalars.

The direct and DeltaRule rows receive the **same** encoded route.  For DeltaRule
the route is L2-normalized before use as a key/query; no separately trained
vector-key encoder is allowed in this ablation.

## Frozen route transforms

Let `r` be the eight-way soft route after the maintained explicit corruption.
Physical slots are partitioned into four contiguous two-slot blocks.  No
semantic labels or test result may be used to choose the partition.

1. `dense_soft`: use all eight components of `r`.
2. `block_top1`: sum `r` within each block, select the largest block, mask all
   other components, and renormalize on the probability simplex.  Fine access
   is therefore two of eight slots.
3. `hard_top1`: retain only `argmax(r)`.  Fine access is one of eight slots.

The selected block is derived from the same route scores.  There is no
auxiliary selector, selector loss, or additional parameter.  Ties use PyTorch's
deterministic first-index rule.

Each transform is crossed with:

- componentwise overwrite/read on direct slots;
- the exact corrective DeltaRule update/read.

Oracle one-hot direct slots and DeltaRule are retained as correctness ceilings.
Triality-bound slots are not repeated in the full grid because the maintained
gauge theorem and 10-seed campaign already make their equality to direct slots
load-bearing; one focused transformed-route replay must check that equality.

## Evaluation grid

Frozen reliability cohort: seeds `0` through `9`.  Seed `101` is development
only and excluded from reliability counts.

The overwrite frontier fixes depth `16`, test radius `0.75`, batch size `32`,
and crosses route perturbation norms `0.00`, `0.10`, `0.20` with supplied
noncommuting transport off/on.

The long-stream frontier uses lengths `256`, `1024`, and `4096`, radius `0.75`,
batch size `8`, perturbations `0.00`, `0.10`, `0.20`, and transport off/on.
The first eight writes initialize all keys.  Later writes target four hot keys;
queries cover hot and cold keys.  Positions before 32 remain warm-up.  Hot,
cold, and combined cohorts are reported separately.

All variants share event schedules, aliases, values, actions, and corruption
draws within a cell.

## Metrics and gates

Report mean/minimum cosine, mean/maximum relative squared error, query count,
selected slots per access, selected fraction, and an ideal payload-byte count
for float32 eight-vector slots.  The byte count excludes routing and launch
overheads and must not be presented as measured bandwidth or latency.

Implementation gates:

- every transformed route is nonnegative and sums to one;
- selected counts are exactly `8`, `2`, and `1`;
- hard one-hot direct and DeltaRule predictions agree to `1e-10` in float64;
- oracle direct and DeltaRule rows are exact to cosine `1 - 1e-10`;
- focused direct/triality transformed-route predictions agree to `1e-10`;
- chunkwise and recurrent DeltaRule states agree to `1e-9`.

No quality winner is declared without all gates passing.  A routing gain is not
a Spin(8)-specific gain because the hierarchy itself uses no triality data.

## Frozen interpretation

The hierarchy is useful if `block_top1` or `hard_top1` improves held-out
retrieval over `dense_soft` while reducing selected payload.  If direct and
DeltaRule converge under hard routing, the result supports the existing
address-bottleneck diagnosis.  It does not establish a new overwrite law.

Because eight slots are too small for kernel-launch savings to dominate, this
quality experiment makes no CUDA speed claim.  Any systems claim requires a
separate large-slot gathered/fused implementation and measured end-to-end
timing.

## Post-primary adversarial frontier frozen before inspection

The completed primary artifacts showed no hard-decision flips through route
perturbation `0.20`.  This is unsurprising because the maintained alias world's
nuisance is constructed orthogonal to its semantic-center span.  Before
running any stronger-corruption cell, freeze an explicitly exploratory
frontier at perturbations `0.30`, `0.50`, `0.75`, and `1.00`, overwrite depth
`16`, stream length `256`, radius `0.75`, transport off/on, and seeds `0` to
`9`.  Training and all other settings remain unchanged.

This secondary screen locates selector failure rather than estimating natural
noise robustness.  It cannot replace or tune the primary grid, and it cannot
support a real-data robustness claim.  Report decision failures and degradation
directly even if they erase the apparent hard-routing advantage.
