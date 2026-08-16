# Frozen-checkpoint `2.A5` unseen-relation results

**Status:** completed exploratory replication; no retraining
**Selection frozen:** 2026-08-16 18:53:01 +02:00
**Evaluation completed:** 2026-08-16 18:56:42 +02:00
**Protocol:**
[`PURE_ROTOR_2A5_UNSEEN_RELATION_PROTOCOL.md`](PURE_ROTOR_2A5_UNSEEN_RELATION_PROTOCOL.md)
**Artifact:**
[`artifacts/pure_rotor_2a5_unseen_relation_exploratory.json`](artifacts/pure_rotor_2a5_unseen_relation_exploratory.json)
**Artifact SHA-256:**
`6580fb4a27cf9a77d19b39ba95fa614026903e81558a605d2e8dcd37c85a3b81`

## Outcome

The original Spin quaternion checkpoints generalize to a second, complete-word
central relation that never occurs in any of the three training schedules. No
model received an optimizer update for this evaluation.

The Spin scan is 100% exact on both early and late length-16 splits for all
three seeds. Its target-versus-central-partner margin remains **100% at L16,
L64, and L128 in every seed**, passing the interpretation frozen before
checkpoint loading. Mean exact accuracy is `76.36%` at L64 and `59.68%` at
L128. Every other trained candidate is near chance on the long center metrics.

This substantially weakens the explanation that the pilot merely memorized
the local rewrite `a,a=z`. It does not prove arbitrary relation generalization:
the deterministic word search and this evaluation were designed after the
original pilot and are therefore explicitly exploratory.

## Unseen relation certificate

The selector reconstructed byte-identical original training schedules and
verified their stored SHA-256 values. It enumerated words by increasing length
and lexicographic token order, rejecting any word that occurred in any seed or
contained `a,a`, `b,b_inverse`, or `b_inverse,b`. Length 11 was the first length
with both an identity-product and center-product survivor:

```text
identity e: a b b b a b b b b b b
center z:   a b b a b b b a b_inverse b_inverse a
```

The exact 120-element multiplication table proves the products are `e` and
`z`. Each complete word has zero occurrences in every seed's realized training
inputs. Every early/late paired schedule passes exact central-partner and
projective-equality audits and carries its own SHA-256.

All oracle contracts pass on every split:

- exact table and float64 quaternion: 100% exact/projective/center/margin;
- projective A5: 100% projective and 50% exact/center/margin.

All 12 source checkpoint hashes were recomputed before evaluation and match the
authoritative pilot artifact.

## Metrics

Values are mean ± sample standard deviation over the same three frozen seeds,
in percent. Tables report the early-relation post-block positions.

### Length 16

| candidate | exact 120 | projective 60 | center bit | central margin |
|---|---:|---:|---:|---:|
| Pure Rotor | 10.94 ± 5.54 | 20.36 ± 6.26 | 45.57 ± 18.93 | 49.52 ± 4.11 |
| identity ablation | 4.38 ± 1.08 | 17.97 ± 4.58 | 41.75 ± 9.54 | 33.38 ± 0.78 |
| **Spin quaternion scan** | **100.00 ± 0.00** | **100.00 ± 0.00** | **100.00 ± 0.00** | **100.00 ± 0.00** |
| Transformers Mamba-2 | 1.43 ± 1.16 | 9.42 ± 10.36 | 40.84 ± 4.26 | 40.06 ± 10.00 |

The late-L16 acquisition split is also 100% on every reported Spin metric and
every seed.

### Length 64

| candidate | exact 120 | projective 60 | center bit | central margin |
|---|---:|---:|---:|---:|
| Pure Rotor | 3.80 ± 0.78 | 7.29 ± 2.18 | 49.50 ± 1.80 | 50.00 ± 0.29 |
| identity ablation | 1.53 ± 0.05 | 4.02 ± 0.65 | 49.53 ± 1.32 | 48.26 ± 0.31 |
| **Spin quaternion scan** | **76.36 ± 6.04** | **76.36 ± 6.04** | **86.93 ± 3.64** | **100.00 ± 0.00** |
| Transformers Mamba-2 | 1.80 ± 0.19 | 4.37 ± 1.21 | 48.91 ± 0.66 | 48.68 ± 1.18 |

### Length 128

| candidate | exact 120 | projective 60 | center bit | central margin |
|---|---:|---:|---:|---:|
| Pure Rotor | 2.20 ± 0.26 | 4.31 ± 0.94 | 49.76 ± 0.92 | 49.88 ± 0.09 |
| identity ablation | 1.13 ± 0.10 | 2.67 ± 0.50 | 49.54 ± 0.41 | 49.19 ± 0.26 |
| **Spin quaternion scan** | **59.68 ± 0.92** | **59.68 ± 0.92** | **78.01 ± 1.05** | **100.00 ± 0.00** |
| Transformers Mamba-2 | 1.17 ± 0.20 | 2.74 ± 0.50 | 49.54 ± 0.45 | 49.55 ± 0.47 |

The Spin per-seed `exact / center / margin` values are:

| seed | L16 | L64 | L128 |
|---:|---:|---:|---:|
| 0 | 100.00 / 100.00 / 100.00 | 82.93 / 90.60 / 100.00 | 60.73 / 79.19 / 100.00 |
| 1 | 100.00 / 100.00 / 100.00 | 75.12 / 86.86 / 100.00 | 59.12 / 77.64 / 100.00 |
| 2 | 100.00 / 100.00 / 100.00 | 71.05 / 83.32 / 100.00 | 59.18 / 77.20 / 100.00 |

Paired final exact accuracy for the Spin model is 100% at L16,
`42.19 ± 8.12%` at L64, and `26.04 ± 0.90%` at L128. As in the original pilot,
perfect central preference is conditional: long-horizon errors increasingly
come from losing the projective product, not from confusing its two lifts.

## Scope and next decision

The empirical statement is narrow: one deterministic unseen relation pair,
one frozen 12-checkpoint cohort, and identical symbolic contexts. It is not a
preregistered result, general language quality, or a theorem about the learned
representation.

The next experiment should not manufacture another favorable word after seeing
these outputs. Freeze a relation family in advance—separate `a^2`, `b^3`, and
`(ab)^5` holds, randomized generator conjugates, and multiple unseen central/
identity words—then retrain and evaluate it against official DeltaProduct or
PD-SSM as well as Mamba-2. That is the correct gate before integrating the Spin
lane into a maintained hybrid.

Executable evidence:

- [`evaluate_pure_rotor_2a5_unseen_relation.py`](../evaluate_pure_rotor_2a5_unseen_relation.py)
- [`test_pure_rotor_2a5_unseen_relation.py`](../test_pure_rotor_2a5_unseen_relation.py)
- [`pure_rotor_ssm/spin_scan.py`](../pure_rotor_ssm/spin_scan.py)
