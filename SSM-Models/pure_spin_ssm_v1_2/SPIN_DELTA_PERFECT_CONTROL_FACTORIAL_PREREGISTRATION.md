# Spin-Delta perfect-control 3x3 factorial

**Protocol status:** specified before all nine outcome runs.

## Question

Perfect causal routing is now established, yet a pristine core still missed a
frozen threshold in one seed. Previous experiments confounded parameter
initialization with minibatch order. This audit crosses three fresh core seeds
with three fresh data-order seeds while supplying exact causal controls from
the first step.

## Frozen design

- initialization seeds: 491, 499, 503;
- data-order seeds: 509, 521, 523;
- all nine Cartesian-product cells;
- one fixed evaluation set shared by every cell;
- exact write/query events and slots supplied causally;
- 800 AdamW steps, batch 128, 16 evaluation batches per length;
- train at 8 writes; evaluate at 8, 16, and 32 writes;
- unchanged `d_model=64`, two-layer Spin-Delta model and raw-CUDA recurrence.

The nine cells may be run in any order, but every implementation hash and
configuration other than the two crossed seeds must match.

## Frozen decisions

**Perfect-control robustness passes** only if every cell reaches at least 95%
at 8 and 16 writes and at least 93% at 32 writes.

**Initialization sensitivity is detected** if, for any fixed data seed, the
16-write accuracy range across initializations is at least five points.

**Data-order sensitivity is detected** if, for any fixed initialization, the
16-write accuracy range across data seeds is at least five points.

These are finite factorial decisions, not population estimates. Failure to
detect a five-point effect does not prove invariance.

## Non-claims

Exact controls remain privileged synthetic information. The audit estimates
optimization sensitivity on one small task and hardware/software stack; it is
not language quality, a general convergence theorem, or a model comparison.
