# Static isotypic retention spectrum: frozen quality gate

**Frozen:** 2026-08-22 before any natural-data training of this candidate.

## Hypothesis

Dynamic tokenwise sector offsets improved two of three frozen seeds but had a
high-variance adverse third seed and failed its mean gate. The useful part may
be distinct triality timescales, not asking each token to synthesize those
timescales from scratch.

Write the maintained shared retention as

```text
s_c(x) = exp(-Delta_c(x)),  Delta_c(x) > 0.
```

The candidate learns one positive static decay rate per inequivalent sector:

```text
lambda_c,r = exp(a_c,r - mean_q a_c,q),
s_c,r(x) = exp(-lambda_c,r Delta_c(x)) = s_c(x) ** lambda_c,r.
```

Centering the log rates fixes their geometric mean at one and keeps the
maintained controller responsible for the common step size. At `a=0`, every
`lambda=1`; the implementation is arranged as a multiplicative residual so
all common parameters and initial logits are bitwise equal to maintained
v1.2. The learned spectrum adds only six scalars per layer, 24 total:
626,540 parameters versus 626,516 maintained and 623,740 for matched Mamba-2.

This is the real isotypic analogue of a selective continuous-time SSM: token
content chooses the step, while learned state modes set their decay spectrum.
It is not a new Spin transport, readout feature, or cross-channel coupling.

## Preserved contracts

- `8v`, `8+`, and `8-` retain independent positive scalar decays, so Spin(8)
  equivariance is preserved;
- every sector retention lies in `(0,1)` and its write is scaled by
  `1-s_c,r`, preserving the per-sector bounded-state argument;
- both independently controlled Spin trajectories and the 48-scalar recurrent
  state are unchanged;
- the existing representation-split raw-CUDA forward/backward consumes the
  resulting `(B,L,C,3)` scale tensor without a new recurrent kernel;
- zero-start, nonzero rate gradients, semantic scan parity, and full-model CUDA
  gradient parity pass before this document is frozen.

## Frozen Tiny Shakespeare gate

- pinned raw-byte Tiny Shakespeare, chronological 90/5/5 split;
- variants: maintained `shared_retention`, then `isotypic_spectrum`;
- fresh seeds: 283, 293, and 307;
- 300 AdamW updates, batch 8, sequence length 256;
- 16 fixed validation batches per seed;
- `d_model=128`, four layers, two channels, `Spin(3,4,6,8)` ladder;
- `raw_cuda_hybrid`, direction readout, SwiGLU;
- learning rate `3e-3`, weight decay `0.01`, clip norm `1.0`;
- identical training and validation batches within each pair.

Positive improvement is `shared_bpb - spectrum_bpb`. Promotion requires all:

1. at least two of three wins;
2. mean improvement at least `+0.0100` bpb;
3. no individual regression worse than `-0.0500` bpb;
4. finite compatible artifacts with bitwise-equal common parameters and
   initial logits.

Only a quality pass authorizes order-balanced complete-step timing. A failure
closes static sector timescales as the immediate successor. Dynamic-retention
seeds 271/277/281 are discovery context only and cannot count toward this gate.

## Claim boundary

A pass would be short-budget evidence for the retention spectrum on this model
and dataset. It would not prove an optimal spectrum, beat Mamba-2, or justify
dense delta memory without a separate matched test.
