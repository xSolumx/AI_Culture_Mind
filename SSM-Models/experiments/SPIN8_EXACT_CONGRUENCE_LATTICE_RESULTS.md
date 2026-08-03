# Spin(8) exact recovered-action congruence lattice: results

Date completed: 2026-08-03.

## Result

The exhaustive audit passed every frozen check. For each prospective seed
49--57 it enumerated all Bell(8)=4,140 set partitions of the recovered anonymous
eight-state action and found exactly six transition congruences:

```text
block count 1: 1 congruence
block count 2: 3 congruences
block count 4: 1 congruence
block count 8: 1 congruence
```

All six quotient actions are regular, exactly matching the six normal subgroups
of Q8. This is invariant to the anonymous cluster labels and holds in 9/9.

## Correction to the preceding claim

The `k=2..12` K-means scan was not a complete congruence-lattice enumeration.
It returned `[8]` in seeds 49 and 53 and `[2,8]` in the other seven seeds. It
therefore missed the four-state quotient in all nine seeds, all two-state
quotients in two seeds, and two of the three distinct two-state congruences even
when cardinality two was found.

The successful empirical result remains intact: the scan selected an eight-state
action that compiled exactly and passed every behavioral gate. What is withdrawn
is the stronger interpretation that the scan proved a unique finest meaningful
congruence of the continuous state system.

## Exact boundary

Transition closure by itself cannot choose a nontrivial semantic quotient. The
universal one-block partition and the discrete partition are congruences of
every total deterministic action. Standard partition refinement needs an
initial observation partition; metric neighborhoods introduce a metric-scale
prior rather than eliminating priors.

The defensible description is now:

> decoder-blind, cardinality-selected metric extraction followed by exhaustive
> certification of the recovered finite action's complete congruence lattice.

Artifacts:

- `spin8_exact_congruence_lattice_seeds49_57.json`
- `spin8_exact_congruence_lattice_seeds49_57.stdout.log`
