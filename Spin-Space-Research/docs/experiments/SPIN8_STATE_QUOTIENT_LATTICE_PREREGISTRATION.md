# Spin(8) state-quotient lattice audit

Date fixed: 2026-08-03, after the cardinality scan found viable sets `[8]`
for seed 38 and `[2, 8]` for seeds 43 and 46.

This is an explanatory audit on already-observed seeds. It cannot repair the
frozen 7/9 state-only gate and it is not prospective reliability evidence.

## Hypothesis

The two-state action in seeds 43 and 46 is not an unrelated alternative
clustering. It is a strict deterministic quotient of the recovered
eight-state action. In automata language, the state geometry admits multiple
stable congruences; the latent action is the finest independently reproducible
congruence, while the two-state partition records token-word parity.

## Frozen checks

For the primary and independent audit corpora of seeds 38, 43, and 46:

1. refit only `k=2` and `k=8` with the already-frozen deterministic eight-
   restart algorithm;
2. map each eight-state cluster to its majority two-state cluster;
3. require every eight-state cluster to have purity at least 0.99 and both
   two-state clusters to have nonempty preimages;
4. require the map to intertwine every recovered token transition exactly;
5. require the primary and audit quotient maps to agree after independently
   aligning their two-state and eight-state centroids;
6. report the sizes of the two quotient fibres and whether all four input
   tokens act by the nonidentity permutation on the quotient.

The hypothesis passes a seed only if `k=2` is structurally viable and all six
checks pass. Seed 38 is the negative reference: its previously measured
`k=2` action is not viable, so it must not be relabeled a quotient success.

## Consequence boundary

If the hypothesis passes seeds 43 and 46, a future cardinality-free compiler
may prospectively select the unique **finest** replicated regular congruence:
all other viable candidates must be exact homomorphic quotients of it. Merely
selecting the numerically largest viable `k` is not an acceptable rule.
