# Learned-router curriculum transfer commissioning failure

**Decision:** the first nine-cell cohort is invalid under its frozen pairing
contract; no retrieval outcome from it is promoted or interpreted.

## Failed provenance gate

The committed protocol required the post-router model digest to be identical
across all three core data-order cells for a fixed initialization seed. The
core was bitwise untouched and identical, and every measured router readiness
metric was 1.0, but independently repeated CUDA router training produced a
different router digest in every cell:

| Init | Core digest prefix | Router digest prefixes across data seeds |
|---:|---|---|
| 587 | `933909260dda` | `20ad73b7e825`, `0d614fd93d36`, `f40758571fc2` |
| 593 | `e099b7e8071e` | `caf31a6c5caa`, `e5b138dde425`, `e35d76149774` |
| 599 | `52f364bdcddc` | `f238ee016dda`, `fb48968f21a7`, `51f9ef057063` |

The runner reset Python, NumPy, CPU Torch, and CUDA seeds, but did not request
globally deterministic CUDA algorithms. The evidence establishes bitwise
nondeterministic router optimization in this execution route; it does not by
itself identify which CUDA backward operation contributed the first differing
bit.

The summarizer stopped before producing a decision. Final retrieval fields
exist in the cell artifacts because each cell is atomic, but they were not
inspected to redesign thresholds or select seeds.

## Structural repair

The replacement protocol does not weaken equality to numerical closeness and
does not rely on independently reproducing CUDA training. It trains one router
exactly once per initialization, holds that checkpoint in memory, and clones
the same bitwise state into all six `(data order, core schedule)` descendants.
Fresh initialization and data seeds prevent the invalid cohort from becoming
development data.

The nine invalid artifacts are retained under
[`artifacts/spin_delta_router_curriculum_transfer_invalid_v1/`](artifacts/spin_delta_router_curriculum_transfer_invalid_v1/).
