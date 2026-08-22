# Spin-Delta: frozen quality gate

**Frozen:** 2026-08-22 after semantic and raw-CUDA parity gates, before any
natural-data training of Spin-Delta.

## Hypothesis

The closed v1.2 controls show that independently controlled Spin transport is
useful, while cross-head mixing and refined triality retention do not explain
the remaining quality gap. The missing operation may instead be addressable
overwrite: maintained v1.2 transports and accumulates, but cannot select a
memory slot to erase, write, and query.

Spin-Delta keeps both independent transport heads and gives each two triality
slots. The homogeneous update is contractive and its address-space action is
rank one. This adds 4,160 controller parameters to the four-layer 128-wide
model and doubles only the Spin recurrent state from 192 to 384 scalars. The
existing convolution history remains 1,536 scalars, so total streaming state
is 1,920 scalars.

## Closed commissioning gates

- associative affine composition and recurrent/parallel full-gradient parity;
- algebraically exact maintained-model embedding with bitwise-identical common
  parameters;
- immediate gradients into write address, erase direction, and query, plus a
  staged erase-strength activation test;
- contractive-left and 384-token causal/finite-state falsifiers;
- raw-CUDA semantic and full-gradient parity at factor counts 3/6/15/28;
- raw-CUDA full-model gradient parity for Spin(3), Spin(4), Spin(6), and
  Spin(8) layers;
- a two-update RTX 2070 SUPER Tiny Shakespeare commissioning smoke.

The generic two-slot float32 contraction has a different reduction shape from
the maintained scalar recurrence. On the frozen full-size `B=8`, `L=256`
pairing probe, the maximum initial-logit residual was
`8.344650268554688e-07`. Therefore this gate freezes a `1e-6` maximum pairing
bound rather than falsely requiring bitwise-identical floating-point logits.

## Frozen Tiny Shakespeare gate

- pinned raw-byte Tiny Shakespeare, chronological 90/5/5 split;
- variants in fixed order: maintained `independent_v1_2`, then `spin_delta`;
- fresh seeds: 353, 359, and 367;
- 300 AdamW updates, batch 8, sequence length 256;
- 16 fixed validation batches per seed;
- `d_model=128`, four layers, two transport heads, two slots per head;
- `Spin(3,4,6,8)` ladder, direction readout, SwiGLU;
- maintained backend `raw_cuda_hybrid`; candidate backend `raw_cuda_delta`;
- learning rate `3e-3`, weight decay `0.01`, clip norm `1.0`;
- identical training and validation batches within each pair.

Positive improvement is `independent_v1_2_bpb - spin_delta_bpb`. Promotion
requires all:

1. at least two of three wins;
2. mean improvement at least `+0.0100` bpb;
3. no individual regression worse than `-0.0500` bpb;
4. finite compatible artifacts, bitwise-equal common parameters, and maximum
   initial-logit difference at most `1e-6`.

Only a quality pass authorizes an order-balanced complete-step speed gate. A
quality failure leaves the CUDA compiler as reusable machinery but closes this
two-head/two-slot parameterization as the immediate successor.

## Claim boundary

A pass is short-budget evidence on one small byte-level language model and one
dataset. It does not establish optimal slot count, scaling, Mamba superiority,
or a Tensor-Core advantage. A failure does not falsify delta memory generally;
it falsifies this frozen Spin-Delta construction and budget.
