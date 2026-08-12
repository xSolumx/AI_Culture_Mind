# Large-slot semantic hierarchy preregistration

- **Frozen:** 2026-08-10, before implementation or result inspection
- **Development seed:** `103`, excluded from every decision
- **Frozen reliability seeds:** `30` through `39`
- **Canonical device/dtype:** CPU `float64`
- **Purpose:** replace the separable eight-slot alias world with overlapping
  semantics and test shared three-view routing independently of the memory law

## Scientific questions

1. Does a learned coarse-to-fine router reduce long-stream interference when
   64 fine keys overlap inside eight semantic blocks?
2. Does one router shared across the three Spin(8) views complete missing
   label/view combinations that three independent, parameter-richer routers do
   not?
3. Do direct overwrite and standard delta memory remain exactly equal when the
   common learned route is made one-hot?

The experiment does not test language modelling, representation discovery, or
extra storage capacity from triality.

## Frozen world

- `64` semantic keys in `8` physical blocks of `8` slots;
- `8`-dimensional canonical aliases and `8`-dimensional values;
- within each block, fine centers are correlated perturbations of one block
  prototype rather than orthogonal keys;
- alias radii: `0.20`, `0.40`, and `0.60`;
- stream checkpoints: `256`, `1024`, and `2048`;
- recurrent batch: `8`;
- three triality views and four action words from one sampled Spin(8) teacher;
- action words `0`, `1`, and `2` are used in router training; word `3` is held
  out until evaluation.

For action word `w` and view `r`, a canonical alias `z` is presented as

\[
x_{w,r}=R_{w,r}z.
\]

Both router families receive the same supplied inverse-frame canonicalization
`R_{w,r}^T x_{w,r}`. This isolates router sharing from action estimation. A
positive result is therefore a shared-family routing result, not evidence that
the Spin(8) action was discovered.

## Missing-view design

Semantic label `k` is omitted from training view `k mod 3`. The union of the
three views contains every label, but each independent router is missing about
one third of its label/view combinations. The shared router pools all supplied
examples. Both families see the same examples, batches, action words, alias
radii, optimizer steps, and losses.

The shared router has one coarse and one fine weight table. The independent
control has three such pairs and therefore three times as many router
parameters. This parameter-richer negative control prevents a positive shared
result from being explained by raw parameter count.

## Frozen training

- normalized linear coarse and fine classifiers;
- loss: global 64-way fine cross-entropy plus eight-way block cross-entropy;
- Adam, `800` steps, batch `512`, learning rate `0.03`;
- fixed softmax temperature `0.35` at evaluation;
- training radii cycle through `0.10`, `0.30`, and `0.50`;
- no seed-specific early stopping or hyperparameter changes.

Every artifact must retain both router weight tables, sampled semantic centers,
block prototypes, Spin(8) action matrices, training losses, and software
versions.

## Frozen routing rows

For both shared and independent routers:

1. `dense_soft`: one softmax over all 64 fine slots;
2. `block_top1`: learned coarse argmax followed by an eight-way softmax inside
   the selected physical block;
3. `hard_top1`: the same learned block followed by a one-hot fine argmax;
4. `oracle`: the semantic label supplied as a one-hot route.

Direct and standard delta memories consume identical routes, values, event
streams, and queries. Values are written only from an observed view. Query
cohorts are reported separately for observed and omitted label/view pairs.

## Frozen implementation gates

Every reliability seed must satisfy:

- maximum supplied inverse-frame canonicalization error `<= 1e-10`;
- maximum hard-route direct/delta state or prediction error `<= 1e-10`;
- oracle minimum query cosine `>= 1 - 1e-10`;
- all metrics finite;
- retained-weight replay difference `<= 1e-12`.

Failure of any implementation gate rejects that seed from scientific
interpretation but remains in the aggregate.

## Frozen scientific decisions

### Shared-router completion

Supported only if at least `8/10` seeds satisfy all of:

- shared held-out-view hard-route accuracy `>= 0.85`;
- shared minus independent held-out-view hard-route accuracy `>= 0.25`;
- independent observed-view hard-route accuracy `>= 0.85`.

### Hierarchical routing

Supported only if at least `8/10` seeds show positive paired mean-cosine
improvement of `block_top1` over `dense_soft` for both direct and delta memory
on the radius `0.20` and `0.40`, length-2048 cells. The full high-noise radius
`0.60` frontier is reported even if the learned block selector fails there.

### Memory-law boundary

Hard direct/delta equality supports only update-law equivalence under a common
one-hot route. Soft direct/delta differences are reported and are not a
triality effect.

## Required claim boundary

A positive result may support learned semantic hierarchy and shared cross-view
routing on this frozen synthetic world. It may not be reported as:

- extra Spin(8) memory capacity;
- a triality-specific advantage over every equivariant prior;
- learned action discovery;
- production throughput or model-level quality.

