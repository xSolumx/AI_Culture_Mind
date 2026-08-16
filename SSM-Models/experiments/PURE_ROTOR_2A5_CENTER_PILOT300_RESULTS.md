# Pure Rotor / 2.A5 center-sensitive pilot-300 results

**Status:** completed empirical pilot; exact task/oracle checks passed
**Protocol frozen:** 2026-08-16 18:25:33 +02:00
**Authoritative replay completed:** 2026-08-16 18:39:55 +02:00
**Artifact:**
[`artifacts/pure_rotor_2a5_center_pilot300.json`](artifacts/pure_rotor_2a5_center_pilot300.json)
**Artifact SHA-256:**
`911815d9e104fa08e632161f97f41a966991a9102c70ca65e52a5f07d28d4476`

## Outcome first

The sign-sensitive Spin quaternion composition scan is the only trained
candidate that passes the preregistered central-margin gate at every seed and
retention length. Its margin between the exact target and that target's central
partner is **100% at lengths 16, 64, and 128 for all three seeds**. Mean exact
binary-state accuracy is `99.76%`, `79.60%`, and `62.20%` respectively.

Pure Rotor v2.1, its identity-transport ablation, and Transformers Mamba-2 all
approach chance on the central bit and target-versus-central-partner margin at
long length. The result isolates a real architectural deficiency: the
maintained recurrence transports state using `Ad(q)`, for which `Ad(q)=Ad(-q)`,
whereas the successful layer retains and composes the Spin representative.

This is a strong result for the frozen finite-group mechanism test. It is **not**
a theorem about all Mamba models, a broad language-model result, or evidence
that an eight-lane quaternion scan is the best general sequence architecture.

## Frozen comparison

| trained candidate | parameters | recurrence used in this screen |
|---|---:|---|
| Pure Rotor v2.1 | 29,370 | bounded affine Cl(3,0) state with rotor conjugation |
| identity-rotation ablation | 29,370 | identical model with rotation disabled |
| Spin quaternion scan | 29,592 | eight unit-quaternion lanes, Hamilton-product prefix scan |
| Transformers Mamba-2 | 29,300 | three-layer `Mamba2ForCausalLM` |

The maximum relative parameter gap is `0.9868%`. This is parameter-near, not
state-matched. All candidates see byte-identical precomputed training and paired
evaluation schedules within each seed. The Pure Rotor and identity candidates
also begin with identical functions.

Configuration: three seeds (`0,1,2`), 300 optimizer updates, batch size 16,
training length 16, two evaluation batches of 32 paired examples, and evaluation
lengths `2,16,64,128`. The device was an NVIDIA GeForce RTX 2070 SUPER with
PyTorch `2.12.0+cu130` and Transformers `5.9.0`.

## Exact task and split checks

The target is every ordered prefix product in the binary icosahedral group
`2.A5` of order 120. The exact table satisfies

```text
a^2 = b^3 = (ab)^5 = z,
z^2 = e,
b b^-1 = e.
```

The exact multiplication-table SHA-256 is
`ee26aff5719e54cf28eccc7a0259c79eeb27c7e06bb18134bf7ca910773f58c9`.
Training excludes the token pair `a,a` while exact reachability witnesses and
the realized schedules cover all 120 binary states, all 60 projective A5
states, and both center bits for every seed. Evaluation replaces `a,a=z` by
`b,b^-1=e` inside an otherwise shared context; after the relation the paired
targets are exact central partners with the same A5 projection. Every paired
audit and stored schedule hash passes.

The oracle controls behave exactly as required:

- exact-table and float64 quaternion oracles: 100% exact, projective, center,
  and central-margin accuracy on every split;
- projective A5 oracle: 100% projective accuracy but 50% exact, center, and
  central-margin accuracy.

Thus the center metric is not recoverable from the quotient label alone.

## Empirical metrics

Values are mean ± sample standard deviation over three seeds, in percent. These
are post-relation metrics for the early-relation retention split.

### Length 16

| candidate | exact 120 | projective 60 | center bit | central margin |
|---|---:|---:|---:|---:|
| Pure Rotor | 12.43 ± 0.99 | 20.42 ± 3.34 | 51.22 ± 0.83 | 54.90 ± 6.23 |
| identity ablation | 11.32 ± 2.04 | 14.29 ± 1.67 | 54.27 ± 1.76 | 59.22 ± 3.66 |
| **Spin quaternion scan** | **99.76 ± 0.21** | **99.76 ± 0.21** | **99.90 ± 0.10** | **100.00 ± 0.00** |
| Transformers Mamba-2 | 32.29 ± 5.96 | 38.63 ± 6.69 | 62.69 ± 4.43 | 64.91 ± 5.11 |

### Length 64

| candidate | exact 120 | projective 60 | center bit | central margin |
|---|---:|---:|---:|---:|
| Pure Rotor | 4.25 ± 0.62 | 7.46 ± 1.18 | 50.16 ± 0.22 | 50.86 ± 1.28 |
| identity ablation | 3.13 ± 0.61 | 4.38 ± 0.76 | 50.95 ± 0.68 | 52.11 ± 1.51 |
| **Spin quaternion scan** | **79.60 ± 2.50** | **79.60 ± 2.50** | **88.59 ± 1.27** | **100.00 ± 0.00** |
| Transformers Mamba-2 | 8.41 ± 1.40 | 10.68 ± 1.60 | 52.65 ± 1.30 | 53.10 ± 1.35 |

### Length 128

| candidate | exact 120 | projective 60 | center bit | central margin |
|---|---:|---:|---:|---:|
| Pure Rotor | 2.29 ± 0.39 | 4.08 ± 1.05 | 50.13 ± 0.11 | 50.36 ± 0.67 |
| identity ablation | 1.89 ± 0.27 | 2.89 ± 0.15 | 50.32 ± 0.18 | 50.97 ± 0.57 |
| **Spin quaternion scan** | **62.20 ± 2.36** | **62.20 ± 2.36** | **79.57 ± 1.01** | **100.00 ± 0.00** |
| Transformers Mamba-2 | 4.56 ± 0.73 | 6.12 ± 0.86 | 51.37 ± 0.43 | 51.67 ± 0.86 |

The Spin scan's per-seed `exact / center / margin` values are:

| seed | L16 | L64 | L128 |
|---:|---:|---:|---:|
| 0 | 100.00 / 100.00 / 100.00 | 82.11 / 89.88 / 100.00 | 63.13 / 79.77 / 100.00 |
| 1 | 99.64 / 99.90 / 100.00 | 79.60 / 88.54 / 100.00 | 59.52 / 78.47 / 100.00 |
| 2 | 99.64 / 99.79 / 100.00 | 77.11 / 87.34 / 100.00 | 63.96 / 80.46 / 100.00 |

On the late-relation length-16 acquisition split, the Spin scan reaches
`99.48 ± 0.45%` exact accuracy and `100.00 ± 0.00%` central-margin accuracy.
Its paired final exact accuracy is `95.83 ± 3.61%` at early L16,
`44.79 ± 4.77%` at L64, and `28.65 ± 5.92%` at L128. The latter decline is a
useful falsifier: perfect conditional center preference does not mean the full
projective product remains correct indefinitely.

## Compute measurements

These figures are descriptive for this eager PyTorch implementation and local
GPU; they are not a kernel benchmark.

| candidate | mean final loss | mean train seconds | mean tokens/s | peak CUDA MiB |
|---|---:|---:|---:|---:|
| Pure Rotor | 2.91115 | 32.62 | 2,355 | 29.48 |
| identity ablation | 3.12308 | 32.82 | 2,341 | 29.48 |
| **Spin quaternion scan** | **0.04819** | **4.35** | **17,666** | **21.09** |
| Transformers Mamba-2 | 2.00312 | 17.22 | 4,460 | 961.61 |

The artifact records all 12 checkpoint paths and checkpoint SHA-256 values.
Local validation recomputed every hash successfully after the authoritative
replay.

## Implemented follow-through

The successful operation is now available as the explicitly experimental
PyTorch module
[`pure_rotor_ssm/spin_scan.py`](../pure_rotor_ssm/spin_scan.py):

- compact scalar-first Hamilton product with verified Cl(3,0) conversion;
- recurrent and differentiable Hillis--Steele product scans;
- identity padding and fixed-size streaming cache;
- trainable token-conditioned Spin increments;
- a minimal classifier wrapper;
- forward, first-order-gradient, center-sign, long-norm, cache, mask, and CUDA
  tests.

This does not silently change Pure Rotor v2.1 or its checkpoint contract. The
new layer is the evidence-backed candidate for a future hybrid: retain bounded
affine multivector memory for ordinary content while exposing a separate signed
Spin composition state to the decoder.

## Interpretation against current research

The result is consistent with, but does not prove or instantiate, the 2026
finite-precision theorem that a single input-dependent complex diagonal SSM
cannot track a non-Abelian group. That theorem is scoped to DCD SSMs, so it must
not be quoted as a theorem specifically about this finite Mamba-2 configuration.
[Shakerinava et al., 2026](https://arxiv.org/abs/2603.01959)

Recent state-tracking architectures attack the same limitation with structured
non-diagonal transitions: DeltaProduct uses products of generalized Householder
maps, while PD-SSM uses permutation-diagonal structure.
[DeltaProduct](https://arxiv.org/abs/2502.10297),
[PD-SSM](https://arxiv.org/abs/2509.22284)

## Next falsifiers

1. Compare the exact same frozen `2.A5` schedules against the official
   DeltaProduct implementation and an official or faithful PD-SSM baseline.
   Mamba-2 is a useful deployed-family control, but no longer the strongest
   state-tracking-specific competitor.
2. Hold out `b,b,b=z` and `(ab)^5=z` separately. A layer that only learned the
   local `a,a` rewrite must fail these new splits; a genuine group-state layer
   should preserve their central action.
3. Randomize generator conjugates and train/evaluate presentations. This tests
   representation learning rather than one convenient coordinate choice.
4. Run state-matched and wall-clock-matched sweeps in addition to the existing
   parameter-near comparison.
5. Only after those structural gates pass, test a preregistered hybrid on a real
   long-context task. Do not promote the experimental module to a maintained
   Pure Rotor version from this pilot alone.

**Post-pilot update (2026-08-16 18:56:42 +02:00):** the first no-retraining
relation falsifier is complete. A deterministic length-11 identity/center word
pair absent from all three training schedules retains 100% Spin central margin
through L128 in every seed, with 59.68% mean exact L128 accuracy. This is an
exploratory replication, not a retroactive part of the frozen pilot. See
[`PURE_ROTOR_2A5_UNSEEN_RELATION_RESULTS.md`](PURE_ROTOR_2A5_UNSEEN_RELATION_RESULTS.md).
