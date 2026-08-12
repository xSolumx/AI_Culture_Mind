# Spin(8) state-cardinality and closure audit

Date fixed: 2026-08-03, after the frozen state-only cohort resolved to 7/9
passes. This is an exploratory mechanistic audit, not a retroactive change to
that gate. Seeds 43 and 46 remain failures regardless of this result. Seed 38
is a passing reference; 43 and 46 localize the two geometric rejections.

## Question

Does the frozen centroid-separation floor reject genuinely ambiguous endpoint
geometry, or can algebraic closure identify the correct eight-state action even
when Euclidean within/between separation falls below 2?

## Audit

For each seed and each candidate cardinality `k=2..12`, independently cluster
the same primary and audit state corpora with the frozen eight-restart k-means
algorithm. Do not use decoder outputs or target labels. Report:

- minimum cluster count, within-cluster RMS, minimum centroid separation, and
  their ratio;
- minimum one-token transition winner fraction and vote gap;
- whether all token actions are permutations;
- whether they generate a regular group of order `k`;
- torsor-origin winner fraction and gap from replaying observed words;
- after Hungarian centroid alignment, the fraction of transitions agreeing
  between the independent corpora and whether their origins agree.

No candidate is accepted merely because its Euclidean score is best. A
structurally viable candidate must have transition winner >=0.99, vote gap
>=0.98, permutation columns, regular closure, origin winner >=0.99, origin gap
>=0.98, and exact independent transition/origin agreement. The audit reports
all viable `k`; uniqueness is an outcome, not an assumption.

## Interpretation

- Unique viability at `k=8`, including seeds 43/46, falsifies the necessity of
  the separation-ratio floor and motivates a new prospectively validated
  closure-first compiler. It does not convert the frozen 7/9 result to 9/9.
- Multiple viable cardinalities expose quotient/refinement ambiguity; fixed
  `k=8` remains essential supervision.
- No viable cardinality in 43/46 confirms that their state geometry lacks a
  stable decoder-free finite quotient at this sampling budget.

Any replacement gate requires new untouched seeds and may not reuse seeds
39--47 in its reliability count.
