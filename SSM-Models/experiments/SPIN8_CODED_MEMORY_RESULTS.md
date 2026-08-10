# Spin(8) coded, tight-frame, and dynamic-slot memory results

> **Archived copy.** Current interpretation lives in
> `Spin8-Triality-Research/docs/experiments/SPIN8_SPIN9_HIERARCHICAL_MEMORY_RESULTS.md`:
> same-router direct and triality slots are capacity-equivalent, and the
> supported triality benefit is cross-view completion.

Date executed: 2026-08-03. Protocols were frozen in
[SPIN8_CODED_MEMORY_PREREGISTRATION.md](SPIN8_CODED_MEMORY_PREREGISTRATION.md),
[SPIN8_TIGHT_FRAME_MEMORY_PREREGISTRATION.md](SPIN8_TIGHT_FRAME_MEMORY_PREREGISTRATION.md),
and
[SPIN8_DYNAMIC_SLOT_PREREGISTRATION.md](SPIN8_DYNAMIC_SLOT_PREREGISTRATION.md).
Raw artifact:
[spin8_triality_coded_memory.json](spin8_triality_coded_memory.json).

## The capacity correction

For unit key (p), triality binding is an orthogonal map (M(p):S^-\to V).
Single-pair retrieval is exact, but every wrong pair in an uncoded
superposition contributes a full-norm orthogonal image. In one channel the
observed mean relative squared errors are 1.0, 2.989, 6.953, and 14.755 for
2, 4, 8, and 16 associations, agreeing with the prediction (K-1). The
random-key inner-product value (1/8) is not an attenuation factor for this
bind--unbind operator.

## Exact multiplicity coding

With code columns (c_k\in\mathbb R^H),

[
\widehat n_q=n_q+\sum_{k\ne q}
\langle c_q,c_k\rangle M(p_q)^TM(p_k)n_k.
]

Orthonormal columns therefore cancel every cross-term. Across all
(K\le H\le32) cells, the maximum relative retrieval error was
8.53e-16.

Random unit codes followed the predicted mean squared error ((K-1)/H);
the largest relative discrepancy over the full grid was 0.1903, inside the
frozen 20% gate.

## Optimal overcomplete coding

For unit code columns, the frame-potential bound gives

[
\frac1K\sum_q\sum_{k\ne q}\langle c_q,c_k\rangle^2
\ge \frac{K-H}{H}
]

when (K>H). Unit-norm tight frames attain it. The largest empirical relative
discrepancy from this prediction was 0.0587, and every tested overcomplete
tight frame with (H>1) beat its matched random code.

| Channels (H) | Associations (K) | Random-code MSE | Tight-frame MSE | Optimum | Coherence |
|---:|---:|---:|---:|---:|---:|
| 4 | 5 | 1.0003 | 0.2483 | 0.25 | 0.25 |
| 8 | 9 | 1.0180 | 0.1244 | 0.125 | 0.125 |
| 16 | 17 | 0.9986 | 0.06096 | 0.0625 | 0.0625 |
| 8 | 16 | 1.8713 | 1.0 | 1.0 | 1.0 |

The last row is an important warning: truncated Walsh tight frames attain the
optimal average frame potential while pairing some columns at coherence one.
Regular simplex frames distribute interference uniformly and have better
worst-case behavior. Average optimality is not a worst-case guarantee.

## Dynamic addressed overwrite

The static codebook was extended to a real recurrent memory:

[
M_t[h]=r_t[h]V_tM_{t-1}[h]+B_t[h].
]

A hard zero retention and triality drive replace one addressed slot; untouched
slots undergo the shared Spin(8) transport. Because all retention operators
are diagonal in one fixed orthogonal multiplicity basis, the transition family
is closed under associative composition.

Across a length-128, eight-slot random overwrite/transport stress test:

- transition associativity error: 2.22e-16;
- parallel/recurrent state error: 9.44e-16;
- recurrent versus directly rebuilt symbolic memory: 2.22e-15;
- final retrieval of all current slot values: 1.05e-15.

This is an exact addressed (H)-slot dynamic memory with constant recurrent
state and logarithmic-depth parallel training.

## Dynamic and scan contracts

- shared Spin(8) memory transport versus rebuilding from transported keys and
  values: 8.88e-16 maximum error;
- retrieval after shared transport: 6.66e-16 maximum error;
- length-64, eight-channel generic affine prefix scan versus recurrent
  execution: 2.22e-15 maximum error.

All frozen gates passed.

## Prior art and defensible contribution

[Hiratani and Sompolinsky (2022)](https://arxiv.org/abs/2204.07186) already
develop octonion quadratic binding, analyze its multi-pair capacity, derive
lower bounds, and enlarge the composition layer. Octonion binding and
capacity scaling are therefore not novelty claims here. Tight frames and
their Welch/frame-potential bound are also classical.

[Gated DeltaNet-2 (2026)](https://arxiv.org/abs/2605.22791) already separates
erase and write gates in fixed-state linear attention, and
[Erase-then-Delta Attention (2026)](https://arxiv.org/abs/2606.26560) already
uses independently addressed rank-one erasure with chunkwise parallel
execution. Exact addressed overwrite and parallel recurrent memory are
therefore not standalone novelty claims either. The direct-slot and delta-rule
families are mandatory baselines for the learned-routing gate.

The project-specific construction is the conjunction of:

1. chiral-key, chiral-value, and vector-memory roles tied by the unique
   Spin(8) triality tensor;
2. optimal multiplicity-space coding;
3. exact addressed overwrite in a shared slot basis;
4. shared noncommutative Spin(8) transport of stored associations; and
5. an exactly associative selective affine scan with constant recurrent state.

Whether that conjunction is publishably novel requires a broader literature
review and task-level baselines.

## No-free-lunch boundary

Exact storage of (K) arbitrary 8D values requires at least (H=K) independent
multiplicity dimensions in this linear construction. When (K>H), the rank
bound makes nonzero average interference unavoidable. Addresses and query keys
are supplied in the dynamic-slot gate. Cleanup decoders may trade approximation
and codebook assumptions for capacity, but they do not invalidate this linear
bound.
