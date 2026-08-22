# Isotypic-retention successor: frozen quality gate

**Frozen:** 2026-08-22 before any natural-data training of this candidate.

## Why this mechanism

The accumulated v1.2 and v1.3 evidence does not support adding more transport
algebra to generic text:

- triality amplitude invariants recovered only `+0.00661` mean bpb;
- readout-only multiplicity routing was non-decisive and too late;
- shared Spin actions lost `0.02415` mean bpb;
- free recurrent mixing lost `0.01565` mean bpb;
- retention-scaled recurrent mixing gained `0.01427` mean bpb but failed every
  complete-step systems gate;
- dense F4/E6 transport did not beat identity transport under v1.3's fresh-seed
  natural-text tests.

The maintained v1.2 recurrence nevertheless imposes one retention scalar on
all three inequivalent triality representations. Spin(8) equivariance does not
require that tying. By the real isotypic decomposition, an equivariant scalar
endomorphism may act independently on `8v`, `8+`, and `8-` because they are
inequivalent irreducible modules.

## Candidate

Maintained v1.2 uses, for channel `c`,

```text
s_c(x) = sigmoid(min_logit + u_c(x)).
```

The candidate adds centered representation offsets

```text
delta_c,r(x) = v_c,r(x) - mean_q v_c,q(x),
s_c,r(x) = sigmoid(min_logit + u_c(x) + delta_c,r(x)).
```

and updates each sector by

```text
h'_c,r = s_c,r R_c,r h_c,r
         + (1 - s_c,r) write_c unit_ball(drive_c,r).
```

The two independently learned Spin controllers, triality action, local
convolution, direction readout, SwiGLU, state size, and all other settings are
unchanged. Centering removes the redundant common offset: `u_c` continues to
control the shared timescale while `v_c,r` controls only relative sector
timescales.

The residual controller is exactly zero initialized. Its construction uses an
isolated RNG fork, so every common parameter and the initial logits are
bitwise equal to maintained v1.2. The gate runner refuses any artifact that
violates this pairing contract.

The candidate adds 3,096 parameters: 629,612 versus 626,516 for maintained
v1.2 and 623,740 for the matched Mamba-2 reference. Its complete recurrent
cache remains 1,728 scalars including convolution history.

## Mathematical and implementation gates already passed

- zero residual gives exactly the maintained shared-retention recurrence;
- sector offsets receive nonzero finite gradients at zero initialization;
- independent scalar sector retention is Spin(8)-equivariant;
- the original per-sector bounded-state argument still applies because every
  `s_c,r` lies in `(0,1)` and the drive is scaled by `1-s_c,r`;
- recurrent and chunk-parallel outputs and all gradients agree in float64;
- raw CUDA output and full gradients agree with the semantic recurrence for
  Spin(3), Spin(4), Spin(6), and Spin(8) factor counts;
- the full-model raw-CUDA gradient path agrees with the semantic compiler.

## Frozen Tiny Shakespeare gate

- dataset: pinned raw-byte Tiny Shakespeare, chronological 90/5/5 split;
- variants: maintained `shared_retention` then `isotypic_retention`;
- seeds: 271, 277, and 281;
- 300 AdamW updates, batch 8, length 256, 16 fixed validation batches;
- `d_model=128`, four layers, two channels, group schedule `3,4,6,8`;
- maintained `raw_cuda_hybrid` backend for both variants;
- learning rate `3e-3`, weight decay `0.01`, clip norm `1.0`;
- identical training batches and validation batches within each seed.

Positive improvement means
`shared_retention_bpb - isotypic_retention_bpb`. Promotion requires all of:

1. at least two of three paired wins;
2. mean improvement at least `+0.0100` bpb;
3. no single-seed regression worse than `-0.0500` bpb;
4. finite compatible artifacts and exact initial pairing.

Only a quality pass authorizes an order-balanced complete-step systems gate.
Sequential training timers are diagnostic and cannot promote or reject the
candidate. A quality failure retains the compiler machinery but closes this
mechanism as the immediate v1.2 successor.

## Claim boundary

A pass would establish only a short-budget, small-model empirical result on
the pinned Shakespeare byte task. It would not close the full gap to Mamba-2,
prove an optimal retention decomposition, or justify additional algebraic
transport.
