# Spin-Delta successor design

**Status:** model architecture target derived from closed v1.2/v1.3 gates;
not yet a promoted model or preregistered natural-data claim.

## Reassessed objective

The target is no longer to make every Spin extension pass. It is to close the
validated quality gap to fused Mamba-2 while preserving the properties that
v1.2 actually earns: exact Spin(8) transport, a tiny streaming cache, bounded
dynamics, and a raw-CUDA path suitable for the RTX 2070 SUPER.

The evidence separates two roles that v1.2 currently conflates:

1. **transport heads** provide independently learned Spin trajectories;
2. **memory slots** provide content-addressed overwrite and retrieval.

Maintained v1.2 has two independent transport heads but only one implicit slot
per head. Its update is leaky accumulation, not a delta memory. v1.3 has real
erase/write/query memory, but dense exceptional transport did not help generic
text. The successor should combine the demonstrated pieces, not the failed
ones.

## State factorization

Let `g=1,...,G` index independent transport heads, `a=1,...,m` index memory
slots, and `r` index `8v`, `8+`, and `8-`. The state is

```text
H[g,a,r] in V_r.
```

The first candidate fixes `G=2`, preserving the two independently controlled
Spin actions that beat shared-action compression, and `m=2`, introducing one
new addressable slot per head. Its recurrent Spin state is 96 scalars per
layer, 384 across four layers. Including the existing 1,536-scalar convolution
history gives 1,920 streaming scalars—about 41.9 times smaller than the 80,384
scalars recorded for the matched Mamba-2 reference.

## Transition

Each head has its own Spin element, shared only across that head's memory
slots:

```text
transported[g,a,r] = rho_r(g_t[g]) H[g,a,r].
```

Within a head, a rank-one safe delta update acts on the slot axis:

```text
L[g] = (I - beta[g] k[g] k[g]^T) diag(s[g]),
H'[g,:,r] = L[g] transported[g,:,r] + k[g] z[g,r]^T,
y[g,r] = q[g]^T H'[g,:,r].
```

Here `k` and `q` are normalized two-slot keys/queries, `0 <= beta <= 1`, and
`0 < s < 1`. Tying erase to the write key is deliberate in the first gate:
`I-beta k k^T` has spectral norm at most one, so the left update remains
contractive. Independent erase/write keys stay a later falsifier rather than
silently weakening the stability theorem.

The readout concatenates the two queried 24-dimensional triality values and
uses the maintained direction interface initially. Triality invariants remain
an isolated optional control because their previous mean effect was too small.

## Exact compiler structure

For each transport head and triality sector, the local linear map is

```text
A[g,r] = L[g] tensor rho_r(g_t[g]).
```

This is an ordinary affine map on `R^m tensor V_r`; chronological composition
is exactly associative after flattening. Crucially, the two independent Spin
actions live in separate transport heads. Slot mixing never assumes that the
two head actions are shared, avoiding the quality loss established by the
shared-action compression gate.

The existing repo already supplies most machinery:

- v1.3 rank-one delta transition and query semantics;
- v1.2 independent Spin controllers and subgroup ladder;
- the coupled-isotypic `2x2` CUDA forward/backward within one shared-action
  slot bank;
- the isotypic forward schedule and guarded reconstruction lessons;
- exact affine composition and full-gradient semantic oracles.

The required new lowering adds an outer transport-head grid dimension rather
than inventing another recurrence kernel from scratch: one warp owns one
`(batch, head)` two-slot bank across the three triality sectors.

## Exact baseline embedding

The candidate must reduce bitwise to maintained v1.2 at initialization:

- slot zero contains the maintained learned initial state;
- slot one starts at zero;
- the initial query selects slot zero;
- delta erase strength starts at zero;
- maintained drive writes into slot zero;
- every new controller is zero initialized under an RNG fork;
- both independent Spin controllers and every common model tensor are copied
  exactly.

The gate runner must refuse training if common parameters or initial logits
differ.

## Acceptance sequence

1. Semantic recurrent equation and affine composition associativity.
2. Exact baseline embedding plus nonzero gradients into key, query, erase, and
   second-slot write controls.
3. Recurrent/parallel output, state, and full-gradient parity in float64.
4. Boundedness and long-sequence finite-state falsifiers.
5. Raw-CUDA output and full-model gradient parity at factor counts 3/6/15/28.
6. A frozen three-seed, 300-step Shakespeare quality gate against unchanged
   maintained v1.2.
7. Only after a quality pass: order-balanced complete-step timing and memory.
8. Only after both gates: longer-horizon Shakespeare plus matched fused
   Mamba-2 confirmation.

No Tensor-Core or Mamba-superiority claim is attached to the design. The first
question is narrower and decisive: does explicit addressable overwrite recover
materially more than the hundredth-bit effects of transport, readout, and
retention modifications?
