# Spin-Delta exact-control write curriculum

**Protocol status:** frozen before every outcome run.

## Question

The perfect-control 3x3 factorial established sufficient capacity but exposed
5.86--8.54-point initialization sensitivity and sign-changing batch-order
effects. Two independent repository programmes found that short-to-long
composition curricula can turn failed fixed-depth training into robust
solutions. Does the same information homotopy repair Spin-Delta's recurrent
core when routing is exact from optimizer step one?

## Frozen paired design

- initialization seeds: `541`, `547`, `557`;
- data-order seeds: `563`, `569`, `571`;
- all nine Cartesian-product cells;
- two arms per cell, initialized bitwise identically;
- fixed arm: 800 updates at 8 writes;
- curriculum arm: 100 updates at 2 writes, 100 at 3, 200 at 5, and
  400 at 8;
- exact write/query events and slots supplied causally in both arms;
- AdamW, learning rate `3e-3`, weight decay `0.01`, gradient clipping at 1;
- batch 128, `d_model=64`, two Spin(8) layers, raw-CUDA Spin-Delta recurrence;
- one fixed 16-batch evaluation set at 8, 16, and 32 writes, shared by all
  18 runs.

The odd and even curriculum depths prevent the even-only sign ambiguity found
in the octonion curriculum. Both arms receive 800 optimizer steps, 102,400
examples, and the same batch size. The fixed arm sees 2,662,400 input tokens;
the curriculum sees 2,009,600, or 75.48% as many. Token exposure is deliberately
not equalized by changing batch size because that would alter gradient noise.
Consequently a curriculum win is conservative with respect to token count, but
the comparison is not a theorem isolating sequence order from every consequence
of shorter examples.

The initial model-state digest must agree across both arms and all data seeds
for each initialization seed. Implementation hashes, evaluation rows, model
shape, optimizer, steps, batch size, and all non-schedule configuration must
agree.

## Frozen decisions

The curriculum is promoted as a **robust core repair** only if all conditions
hold:

1. every curriculum cell reaches at least 95% at 8 and 16 writes and at least
   93% at 32 writes;
2. the curriculum's worst 16-write cell exceeds the fixed arm's worst cell by
   at least two percentage points;
3. both its maximum fixed-data initialization range and maximum fixed-init
   data-order range at 16 writes are below five points;
4. its mean paired 16-write improvement is nonnegative;
5. no paired 16-write cell regresses by more than two points.

The report also applies the original robustness thresholds to the fresh fixed
arm and records every individual sensitivity range. Passing only a mean or a
single seed is not promotion.

## Interpretation boundaries

This is privileged synthetic supervision and a finite 18-run optimization
experiment. It is not natural-language evidence, a convergence theorem, or a
claim that curricula universally outperform fixed-depth sampling. A pass
authorizes a separate self-supervised or natural-data translation; it does not
silently promote Spin-Delta into maintained v1.2.
