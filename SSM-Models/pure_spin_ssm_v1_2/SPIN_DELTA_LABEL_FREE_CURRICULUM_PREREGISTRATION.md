# Spin-Delta label-free curriculum gate

**Protocol status:** frozen before every outcome run.

## Question

Supervised commissioning plus the 2/3/5/8 information homotopy makes the
learned-router Spin-Delta core robust. Can the router and core discover the same
causal computation jointly from retrieval labels alone?

## Frozen design

- initialization seeds: `653`, `659`, `661`;
- data-order seeds: `673`, `677`, `683`;
- one initial model is constructed once per initialization and cloned into all
  six `(data order, schedule)` descendants;
- fixed arm: 800 joint router/core updates at 8 writes;
- curriculum arm: 100 joint updates at 2 writes, 100 at 3, 200 at 5, and
  400 at 8;
- the only differentiated objective is final retrieval cross-entropy;
- no router labels, router auxiliary loss, oracle controls, teacher controls,
  or frozen router are used;
- grammar labels are retained only for detached measurement of router behavior;
- fixed token-only evaluation at 2, 3, 5, 8, 16, and 32 writes;
- unchanged batch 128, AdamW `3e-3`, weight decay `0.01`, clip norm 1,
  `d_model=64`, two Spin(8) layers, and raw-CUDA recurrence.

Both arms use 800 updates and 102,400 retrieval labels. Fixed depth receives
2,662,400 tokens and the curriculum 2,009,600.

## Slot gauge

Without router labels, the names of the two memory slots have a global
permutation symmetry. Swapping both write and query slot identities everywhere
computes the same recurrence. Router identification is therefore evaluated
under one shared binary permutation selected over all readiness lengths for a
cell. Write/query event metrics are not gauge-adjusted. Choosing separate
write and query permutations is forbidden.

## Frozen decisions

**Retrieval autonomy passes** only if the curriculum:

1. reaches at least 95% in every 8/16-write cell and 93% in every 32-write cell;
2. improves the worst 16-write cell over fixed depth by at least two points;
3. keeps both 16-write factorial-range maxima below five points;
4. has nonnegative mean paired 16-write improvement;
5. has no paired 16-write regression greater than two points.

**Router identification passes** only if every curriculum cell, after its one
global slot-gauge choice, reaches at least 0.99 for write-event F1, query-event
F1, write-slot accuracy, and query-slot accuracy at all six evaluation lengths.

**Learning autonomy passes** only if the no-label/no-oracle contract, retrieval
autonomy, and router identification all pass. Retrieval success with a
different latent routing scheme remains a separately reported result rather
than being relabeled as grammar recovery.

## Boundaries

Final retrieval targets remain supervised. This gate removes privileged router
and control labels; it is not unsupervised language modeling, natural-data
quality, a speed result, or a maintained-model promotion.
