# Independent-action block-affine recurrence

**Implementation and gate frozen before training:** 2026-08-22

## The correction that opens this frontier

The shared-action experiment established that v1.2 benefits from two
independently controlled Spin trajectories. Sharing the action is necessary
for the compact factorization `L tensor R` to remain closed under composition,
but it is **not** necessary for global Spin equivariance or finite-dimensional
associative closure.

Let channel `c` carry an independent triality action `R_{t,c,r}` in sector
`r`. Define

\[
\widetilde H_{t,c,r}=R_{t,c,r}H_{t-1,c,r},
\qquad
H_{t,a,r}=\sum_c L_{t,ac}\widetilde H_{t,c,r}+D_{t,a,r},
\]

with

\[
L_t=\operatorname{diag}(s_t)Q_t,
\qquad Q_t\in SO(2).
\]

At `Q_t=I`, this is exactly the maintained independent-action v1.2
recurrence. The optional controller is RNG-neutral and zero-initialized after
the model-wide initializer, so candidate and baseline have exactly equal common
parameter tensors. Tests establish numerical full-model identity and nonzero
finite coupling-controller gradients.

## Equivariance and stability

Under a common frame change `G_r` in each triality sector,

\[
H_{c,r}\mapsto G_rH_{c,r},\quad
R_{c,r}\mapsto G_rR_{c,r}G_r^{-1},\quad
D_{c,r}\mapsto G_rD_{c,r}.
\]

Because the entries of `L_t` are scalars on the representation coordinate,
the whole recurrence transforms by `G_r`. Thus the cross-channel action is
globally equivariant even though it does not commute with the tokenwise block
diagonal matrix of unequal `R_{t,c,r}`.

For one sector, the linear transition is

\[
A_{t,r}=(L_t\otimes I_8)
\operatorname{diag}(R_{t,1,r},R_{t,2,r}).
\]

The right factor and `Q_t tensor I_8` are orthogonal, hence

\[
\|A_{t,r}\|_2=\|\operatorname{diag}(s_t)\|_2
=\max_c s_{t,c}<1.
\]

The established contractive bound therefore survives exactly.

## Associative compiler boundary

Unequal channel actions mean token factorizations of the displayed form are
not generally closed under composition. The exact closure is instead the
ordinary affine monoid on the flattened 16-dimensional channel-state block:

\[
(A_2,D_2)\circ(A_1,D_1)
=(A_2A_1,A_2D_1+D_2).
\]

`independent_block_scan.py` materializes one `16 x 16` operator per triality
sector as the transparent prefix oracle. It is an exact finite-dimensional
compiler, not a claim that dense block composition is the optimal hardware
schedule. The streaming implementation evaluates the structured token factors
directly and retains the original 48-scalar cache.

The raw CUDA lowering assigns one warp to both channels and all three sectors.
It applies both independent coordinate streams, performs the `2 x 2` mixer,
and returns coordinate, mixer, drive, and initial-state gradients. Reverse mode
replays pre-affine states and never inverts the mixer.

Before this freeze, the maintained cu126 WSL suite passed 57 tests, including:

- block-affine associativity;
- recurrent/prefix output and gradient parity;
- exact spectral contraction and identity reduction;
- common-frame equivariance;
- full-model recurrent/prefix parity, causality inherited from the token scan,
  and identity-start controller gradients;
- raw CUDA output and full-gradient parity for 3, 6, 15, and 28 Spin factors;
- full-model semantic/raw-CUDA backward parity.

## Frozen Shakespeare gate

- baseline: maintained independent recurrence with `raw_cuda_hybrid`;
- candidate: `independent_block`, orthogonal recurrent multiplicity, and
  `raw_cuda_block`;
- fresh seeds: `193`, `197`, and `199`;
- Tiny Shakespeare raw UTF-8 bytes, maintained 90/5/5 split;
- 300 steps, batch 8, sequence length 256, 16 fixed validation batches;
- group schedule `(3,4,6,8)`, direction readout, SwiGLU, no readout router;
- paired common initialization, training batches, validation windows,
  optimizer, and WSL PyTorch 2.10/cu126 environment.

The candidate adds one 128-to-1 controller per layer: 627,032 parameters
versus 626,516 for maintained v1.2, an increase of 0.0824%. The official fused
Mamba-2 reference has 623,740 parameters.

Positive improvement is `maintained bpb - independent-block bpb`. Promote to a
speed gate only if all hold:

1. the candidate wins at least two of three seeds;
2. mean improvement is at least `+0.0100` bpb;
3. no seed regresses by more than `0.0500` bpb;
4. all values and gradients are finite and implementation hashes agree.

Sequential training timers are diagnostic only. If quality passes, a separate
order-balanced complete-step campaign must show no more than 10% throughput
regression before changing the maintained default. Failure rejects this
particular `SO(2)` block coupling, not the block-affine theorem or CUDA backend.
