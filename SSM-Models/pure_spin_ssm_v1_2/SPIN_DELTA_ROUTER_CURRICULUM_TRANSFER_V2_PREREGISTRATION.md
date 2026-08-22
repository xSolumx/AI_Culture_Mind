# Spin-Delta learned-router curriculum transfer v2

**Protocol status:** frozen after the v1 commissioning failure and before all
v2 outcome runs.

## Repair and question

The scientific question and promotion thresholds are unchanged from v1. The
only repair is structural pairing: one router is trained once per initialization
and retained in memory while all three data-order cells and both core schedules
are cloned from it.

## Frozen cohort

- initialization seeds: `617`, `619`, `631`;
- core data-order seeds: `641`, `643`, `647`;
- one process and one 100-step, 8-write router training run per initialization;
- three data-order descendants from that exact in-memory router checkpoint;
- fixed core arm: 800 updates at 8 writes;
- curriculum core arm: 100 updates at 2 writes, 100 at 3, 200 at 5, and
  400 at 8;
- unchanged batch 128, AdamW `3e-3`, weight decay `0.01`, clip norm 1,
  `d_model=64`, two Spin(8) layers, and raw-CUDA Spin-Delta recurrence;
- fixed token-only evaluation at 2, 3, 5, 8, 16, and 32 writes;
- retrieval decisions at 8/16/32 and router-readiness decisions at all six
  lengths;
- audit every phase-B hard event and conditional slot decision without using
  those labels in the model forward path or retrieval loss.

All three artifacts for an initialization must carry the same cohort execution
identifier, post-router full-state digest, core digest, router digest, readiness
metrics, and implementation hashes. Both arms within every cell must begin from
that exact state.

## Frozen decisions

Autonomous-router validity requires at least 0.99 on all four readiness metrics
at all six lengths, exactly 1.0 for all four audited phase-B training metrics,
an untouched phase-A core, frozen phase-B router, and bitwise clone equality.

Curriculum transfer additionally requires:

1. every curriculum cell at least 95% at 8/16 writes and 93% at 32;
2. at least a two-point improvement in the worst 16-write cell;
3. both curriculum factorial-range maxima below five points;
4. nonnegative mean paired 16-write improvement;
5. no paired 16-write regression greater than two points.

The fixed and curriculum arms retain identical update/example counts and the
previously frozen 2,662,400 versus 2,009,600 token exposures.

## Boundaries

The v1 artifacts are a provenance failure, not a negative model result. V2
still uses supervised router commissioning and synthetic audit labels. A pass
would establish autonomous forward-control transfer on this grammar, not
self-supervised routing or natural-language quality.
