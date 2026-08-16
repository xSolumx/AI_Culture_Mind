# Spin / DeltaProduct / Mamba-2 multi-relation `2.A5` result

**Protocol frozen:** 2026-08-16T19:20:57+02:00
**Run:** 2026-08-16T19:22:56+02:00 to 2026-08-16T19:29:51+02:00
**Device:** NVIDIA GeForce RTX 2070 SUPER
**Authoritative artifact:**
[`spin_2a5_multirelation_pilot300.json`](artifacts/spin_2a5_multirelation_pilot300.json)
**Artifact SHA-256:**
`054527e8c3e30d64df30217c2128616e82e7f2025278c200cdbde611647fe6d4`

## Outcome

The explicit Spin quaternion product scan passes the preregistered
coordinate-robust center-retention gate. Its worst central-margin accuracy is
**99.50%** and its mean is **99.84%** across all 18 combinations of three
withheld relations, three conjugated generator coordinates, and early L64/L128
retention. The required threshold was greater than 75% in every split.

It also passes the stricter learned-model winner rule: Spin has uniquely highest
early-L64/L128 exact accuracy in **18/18** relation-by-coordinate splits, with
no ties. Pure Rotor v2.1, identity transport, Transformers Mamba-2, and the
DeltaProduct reference all fail the center-retention gate.

This is a strong one-initialization mechanism pilot, not replicated multi-seed
evidence. The three coordinate runs share initialization seed 0 and
byte-identical token schedules; conjugation only changes the exact target
labelling. A fresh frozen multi-seed cohort is required before calling the
result replicated.

## Registered long-retention gate

Values pool the 18 `early_L64`/`early_L128` splits only for summary; the gate
itself was applied to every split independently.

| Candidate | Minimum center margin | Mean center margin | Gate | Unique exact wins |
|---|---:|---:|---|---:|
| Pure Rotor v2.1 | 50.06% | 51.40% | fail | 0/18 |
| Identity ablation | 50.48% | 52.85% | fail | 0/18 |
| **Spin quaternion scan** | **99.50%** | **99.84%** | **pass** | **18/18** |
| Transformers Mamba-2 | 52.44% | 54.84% | fail | 0/18 |
| DeltaProduct reference | 50.65% | 52.20% | fail | 0/18 |

Spin's direct center preference survives even as full-state decoding degrades
with length. This distinction matters: a model can preserve the central bit
while losing other coordinates of the group product.

## Early-relation averages

These are means over all three relations and all three conjugated coordinates.

| Candidate | L16 exact | L16 margin | L64 exact | L64 margin | L128 exact | L128 margin |
|---|---:|---:|---:|---:|---:|---:|
| Pure Rotor v2.1 | 28.29% | 60.08% | 8.59% | 51.98% | 4.57% | 50.83% |
| Identity ablation | 24.39% | 67.24% | 6.08% | 53.81% | 3.22% | 51.88% |
| **Spin quaternion scan** | **99.99%** | **100.00%** | **68.84%** | **99.99%** | **41.23%** | **99.70%** |
| Transformers Mamba-2 | 40.15% | 76.64% | 9.82% | 56.41% | 5.08% | 53.27% |
| DeltaProduct reference | 51.86% | 64.55% | 18.87% | 53.05% | 9.79% | 51.36% |

DeltaProduct is the strongest non-Spin candidate for full exact state at L64
and L128, but its central margin falls toward chance. Under this budget its
matrix fast-weight state learned more of the generic prefix task than Mamba-2
without learning a persistent binary-center mechanism. That is a useful
negative separation, not evidence that DeltaProduct generally cannot represent
the group.

## Spin result by withheld relation

Values are means over the three conjugated coordinates.

| Withheld relation | L16 exact / margin | L64 exact / margin | L128 exact / margin |
|---|---:|---:|---:|
| `a^2=z` | 100.00% / 100.00% | 70.14% / 100.00% | 42.48% / 99.58% |
| `b^3=z` | 100.00% / 100.00% | 75.26% / 100.00% | 43.57% / 99.86% |
| `(ab)^5=z` | 99.96% / 100.00% | 61.12% / 99.96% | 37.64% / 99.67% |

The length-10 relation is harder for complete 120-state decoding but not for
central-sign retention. The result therefore rejects the narrow explanation
that the earlier Spin success was only a memorized `a,a` detector.

Late-block acquisition tells the same qualitative story. Across every
relation, length, and coordinate, Spin's mean late central margin is 99.62%
with a 96.88% minimum. The other candidates average 50.98--54.34% and each has
at least one below-chance late split.

## Exact and provenance audits

- All three training languages exclude every occurrence of `a^2`, `b^3`, and
  `(ab)^5`, yet exact breadth-first search reaches all 120 states within length
  10.
- Realized training covers all 120 binary targets, all 60 projective targets,
  and both center bits.
- Input training and evaluation schedules are byte-identical across the three
  conjugated coordinates.
- Every center evaluation word contains exactly its one designated forced
  relation; every identity mate contains no withheld relation.
- All scored target pairs pass exact central-partner and common-projective-
  label checks.
- All **216** oracle/split contracts pass. Exact table, float64 quaternion, and
  exact regular PD controls are perfect. The projective oracle is exactly 100%
  projective and 50% on exact/center/margin metrics.
- All **15** checkpoint files reproduce their recorded SHA-256 values.
- Pure Rotor and identity-ablation initial functions are identical in all three
  coordinates.
- Learned candidates have 29,288--29,624 parameters, a maximum 1.13% gap.

The result artifact hash and registered gates are locked by
[`test_spin_multirelation_2a5.py`](../test_spin_multirelation_2a5.py).

## Runtime diagnostics, not a systems ranking

| Candidate | Mean train seconds | Mean tokens/s | Peak CUDA MiB |
|---|---:|---:|---:|
| Pure Rotor v2.1 | 32.99 | 2,328 | 74.4 |
| Identity ablation | 32.77 | 2,344 | 74.4 |
| Spin quaternion scan | 4.32 | 17,783 | 68.5 |
| Transformers Mamba-2 | 19.35 | 3,970 | 1,010.0 |
| DeltaProduct reference | 6.42 | 11,961 | 86.5 |

These numbers are local eager/backend diagnostics. Mamba-2 used Transformers'
unfused fallback. DeltaProduct used the equation-faithful unfused PyTorch
reference, not official FLA/Triton kernels. State is not matched: Spin uses 32
real recurrent scalars, Pure Rotor/identity 216, and DeltaProduct 256; Mamba-2
retains its architecture-specific cache. No production throughput conclusion
is licensed.

## What this changes

The most defensible Pure Rotor upgrade is **not** to claim that sandwich
transport has learned the binary center. The result instead supports a
separate sign-sensitive Spin product state as the useful custom primitive:

1. retain the Hamilton-product scan as an explicit experimental layer;
2. use Pure Rotor's bounded affine write/decoder around it only in a declared
   hybrid, rather than conflating `q x q^-1` transport with `q` itself;
3. validate the same frozen architecture on fresh initialization seeds before
   adding capacity, auxiliary losses, or real-sequence tasks;
4. then state-match against 32-scalar structured baselines and profile fused
   kernels separately from learning quality.

## Limits and nonclaims

- One initialization seed across three coordinates is not a three-seed model
  replication.
- Exact oracles establish representability, not learnability.
- Failure under 300 updates is not a theorem about Mamba-2, DeltaProduct,
  PD-SSM, diagonal SSMs, or commutative models.
- The DeltaProduct reference matches the relevant state-tracking equations but
  not the official source training scale or fused implementation.
- This symbolic group benchmark is not a language-model result.
- The Spin scan is a purpose-built noncommutative inductive bias and remains
  separate from canonical Pure Rotor v2.1.
- No result here supplies a geometric Dirac operator, spectrum, zero-mode, or
  index theorem.

## Next frozen falsifier

Without changing the architecture or thresholds, repeat this exact protocol
on fresh seeds `10--19` or a smaller preregistered validation cohort if compute
is constrained. The validation should be judged per relation and seed, retain
the conjugated-coordinate control, and report failures rather than tuning on
them. Only after that cohort should a hybrid Spin-plus-affine-write layer be
designed for non-symbolic sequence tasks.
