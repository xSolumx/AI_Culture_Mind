# Spin(8) finest-congruence compiler: results

Date completed: 2026-08-03.

> **Post-freeze correction, 2026-08-03:** the behavioral 9/9 result below is
> unchanged, but “unique finest congruence” was too strong. Exhaustive
> enumeration later found the complete Q8 congruence histogram
> `{1:1, 2:3, 4:1, 8:1}` in every seed. The K-means scan selected the largest
> replicated metric candidate; it did not enumerate the full lattice or identify
> a semantic quotient without a prior. See
> `SPIN8_EXACT_CONGRUENCE_LATTICE_RESULTS.md`.

## Result: 9/9, every frozen gate passed

The compiler receives recurrent states, uniformly sampled token strings, and
one-token state successors. It receives no decoder/logits, target labels,
state cardinality, Cayley table, inverse pairs, token-to-element map, identity
label, or group-aware sampler during discovery. It scans the coarse bound
`k=2..12` and accepts a unique finest reproducible regular congruence only when
every other viable action is certified as its exact homomorphic quotient.

| Cohort | Passes | Selected cardinality | Dense and L16K |
|---|---:|---:|---:|
| Excluded development seed 43 | 1/1 | 8 from `[2, 8]` | 1/1 |
| Prospective smoke seed 48 | 1/1 | 8 from `[8]` | 1/1 |
| Untouched seeds 49--57 | **9/9** | 8 in 9/9 | **9/9** |

The prospective reliability requirement was at least 8/9. Uniform 9/9 is a
stronger observed result, not a changed gate.

## The actual result

The preceding fixed-cardinality compiler failed its untouched gate at 7/9.
Its Euclidean separation threshold rejected state geometries that still
contained exact finite actions. The replacement does not lower that threshold;
it removes Euclidean separation from acceptance and replaces it with an
algebraic refinement certificate.

In the untouched cohort, seeds 52, 55, and 57 have separation ratios
`1.880197`, `1.892014`, and `1.861115`, all below the old `2.0` floor. All
three now pass because their selected eight-state action is deterministic,
regular, independently replicated, and finer than every other viable action.
Thus the exact intervention repairs three prospective cases the old rule would
have refused.

Across seeds 49--57:

| Metric | Range / count |
|---|---:|
| Selected cardinality | 8 in 9/9 |
| Viable spectrum `[8]` | 2/9 |
| Viable spectrum `[2, 8]` | 7/9 |
| Primary separation ratio | 1.8611--4.3394 |
| Exact section rank | 8 in 9/9 |
| Section condition number | 25.18--129.19 |
| Maximum centroid projection RMS | 0.00439--0.00928 |
| Spin(8) action reconstruction max | 5.03e-7--8.54e-7 |
| Recovered-table homomorphism RMS | 5.70e-7--7.12e-7 |
| Observer transport max | 5.75e-8--9.21e-8 |

Every compiler gate and every post-hoc gate passes. Every recovered anonymous
table is post-hoc Q8-isomorphic, every dense L15--L256 central-pair cell is
100%, and every L4095/L4096/L16383/L16384 cell is 100%. No gradient step is
taken after compilation.

## A repeated learned quotient

Seven of nine untouched seeds independently expose both the full Q8 action and
a balanced two-state quotient. In all seven, the certified map has purity
`1.0`, fibre sizes `[4,4]`, and transition-intertwining fraction `1.0` in both
corpora. Six place the `i/-i` generator pair in the quotient kernel and one
places `j/-j` in the kernel. This is a recurring generator-aligned character
`Q8/C4 ~= C2`, not merely a low-separation failure mode.

The observation is mechanistically useful but should not be romanticized:
endpoint-supervised training created these states, the scan range `2..12` is a
coarse prior, and only Q8 has been validated. The result is cardinality-free
and decoder-free **compilation**, not unsupervised group learning or a language-
model result.

## New boundary

The next falsifier is cross-group. The same finest-congruence rule must recover
a non-Q8 action—preferably one with a different quotient lattice and without a
center-faithful 8D section—without changing the search/gate logic. That test
will separate a general automaton-congruence compiler from a Q8-specialized
success.

Artifacts:

- `spin8_finest_congruence_smoke_seed48_raw.json`
- `spin8_finest_congruence_smoke_seed48_compiled.json`
- `spin8_finest_congruence_validation_seeds49_57_raw.json`
- `spin8_finest_congruence_validation_seeds49_57_compiled.json`
- `spin8_finest_congruence_validation_seed49.json` through `seed57.json`
