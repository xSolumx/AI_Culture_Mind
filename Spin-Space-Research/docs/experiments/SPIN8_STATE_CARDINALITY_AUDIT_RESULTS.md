# Spin(8) state-cardinality audit: results

Date completed: 2026-08-03. This audit is exploratory and leaves the frozen
state-only result at 7/9.

## Result

Candidate cardinalities `k=2..12` were clustered independently on two state
corpora per checkpoint. Structural viability required deterministic token
transitions, permutation actions, regular group closure, a unique word-
consistent torsor origin, and exact transition/origin replication after
geometric alignment.

| Seed | Frozen state-only status | Separation at k=8 | Viable cardinalities |
|---:|---:|---:|---:|
| 38 | pass | 2.489770 | `[8]` |
| 43 | fail | 1.896973 | `[2, 8]` |
| 46 | fail | 1.834542 | `[2, 8]` |

At `k=8`, all three seeds have transition winner fraction `1.0`, transition
gap `1.0`, origin winner/gap `1.0`, regular closure, and exact cross-corpus
agreement. The Euclidean floor therefore rejected two real eight-state
actions.

But closure does not infer cardinality: the two rejected seeds also possess a
replicated deterministic regular two-state action. All `k=3..7` except this
`k=2` case, and all `k=9..12`, fail at least one structural criterion. This is
evidence for nested finite quotients, not permission to select eight because
the task designer expects Q8.

The preregistered quotient-lattice audit tests whether the two-state partition
is an exact homomorphic coarsening of the eight-state action. A future
cardinality-free compiler is justified only if it can identify a unique finest
reproducible congruence and reject incomparable alternatives.
