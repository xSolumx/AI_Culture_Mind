# Spin-Delta learned-router curriculum transfer

**Protocol status:** frozen before every outcome run.

## Question

The exact-control 2/3/5/8-write curriculum repaired the Spin-Delta core across
a fresh 3x3 initialization-by-data-order factorial. The causal router separately
learned every event and slot, but its earlier fixed-depth core remained
seed-unstable. Does the successful information homotopy survive when controls
come autonomously from that learned router rather than from oracle tensors?

## Frozen paired design

- initialization seeds: `587`, `593`, `599`;
- core data-order seeds: `601`, `607`, `613`;
- all nine Cartesian-product cells;
- phase A trains only the causal router for 100 updates at 8 writes;
- the Spin-Delta core must remain bitwise untouched during phase A;
- the trained router is frozen and one post-router model state is cloned
  bitwise into both core-training arms;
- fixed arm: 800 core-only updates at 8 writes;
- curriculum arm: 100 updates at 2 writes, 100 at 3, 200 at 5, and 400 at 8;
- evaluation uses token IDs only; no oracle controls enter the model;
- AdamW, learning rate `3e-3`, weight decay `0.01`, clip norm 1;
- batch 128, `d_model=64`, two Spin(8) layers, raw-CUDA Spin-Delta recurrence;
- fixed evaluation rows at 2, 3, 5, 8, 16, and 32 writes, with retrieval
  decisions scored at 8, 16, and 32.

For a fixed initialization seed, phase-A data are identical across all three
core data seeds, so the post-router model digest must also be identical. Each
core arm begins with a fresh optimizer and the same cloned state. During every
phase-B batch the runner audits the router's hard event and slot decisions
against labels used only for measurement; retrieval training receives no
oracle control tensor or auxiliary router loss.

Both arms receive 800 core updates, batch 128, and 102,400 examples. As in the
successful exact-control gate, the curriculum receives 2,009,600 tokens versus
2,662,400 for fixed depth.

## Frozen decisions

**Autonomous-router validity passes** only if:

1. after phase A, write-event F1, query-event F1, write-slot accuracy, and
   query-slot accuracy are each at least 0.99 at all six evaluation lengths;
2. every hard event and conditional slot decision is correct across every
   phase-B training example in both arms;
3. the core is bitwise untouched in phase A, both arms have the same cloned
   state digest, and the router remains frozen in phase B.

**Curriculum transfer passes** only if autonomous-router validity passes and:

1. every curriculum cell reaches at least 95% at 8 and 16 writes and at least
   93% at 32 writes;
2. its worst 16-write cell exceeds the fixed arm's worst cell by at least two
   percentage points;
3. its maximum fixed-data initialization range and maximum fixed-init
   data-order range at 16 writes are both below five points;
4. mean paired 16-write improvement is nonnegative;
5. no paired 16-write cell regresses by more than two points.

These reproduce the exact-control decision rather than inventing an easier
post-result threshold.

## Boundaries

Router labels remain privileged during the 100-step router pretraining phase
and during the non-differentiated audit. A pass establishes autonomous forward
routing and robust synthetic retrieval after supervised router commissioning;
it is not self-supervised event discovery, natural-language quality, or a
maintained v1.2 promotion.
