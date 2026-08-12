# Spin(8) exact recovered-action congruence-lattice audit: preregistration

Date frozen: 2026-08-03, before executing the exhaustive audit.

## Question

Does the `k=2..12` replicated K-means scan enumerate the complete congruence
lattice of the recovered eight-state action, and can transition closure alone
identify a unique nontrivial quotient without observations?

## Exact prediction

For the regular Q8 action, action congruences correspond to cosets of subgroups.
Q8 has six subgroups: the trivial subgroup, its center, three order-four cyclic
subgroups, and Q8 itself. Therefore exhaustive enumeration of all Bell(8)=4,140
set partitions must return exactly six congruences with block-count histogram:

```text
{1: 1, 2: 3, 4: 1, 8: 1}
```

All six induced quotient actions should be regular because every Q8 subgroup is
normal. The existing metric scan reports only `[8]` or `[2,8]`; it is therefore
predicted to omit the four-state quotient and at least two two-state quotients.

## Identifiability boundary

For every total deterministic action, both the universal one-block partition
and the discrete partition are congruences. Paige-Tarjan refinement from one
unobserved block consequently remains at one block. Initializing it from metric
neighborhoods makes the answer conditional on that metric threshold.

The audit must therefore reject the claim that transition closure alone selects
a unique meaningful nontrivial quotient. This is a theorem-level limitation,
not a failed optimization run.

## Frozen pass conditions

1. Enumerate all 4,140 partitions independently for every prospective seed
   49--57 recovered action.
2. Recover the exact histogram `{1:1, 2:3, 4:1, 8:1}` in every seed.
3. Demonstrate that the metric scan is incomplete in every seed.
4. Keep the successful empirical claim narrow: K-means selected an eight-state
   action that compiles correctly, but it did not enumerate or prove uniqueness
   in the complete congruence lattice.
